# %%
import numpy as np
import pandas as pd
import pickle
import time
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import copy
import math
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score, classification_report
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
#from torchao.prototype.mx_formats.inference_workflow import NVFP4WeightOnlyConfig
import re
from si_prefix import si_format


# %%
########## class definitions ############

# Define Positional Encoding and Transformer Classifier
# It is important that they look the same as their counterparts in the main program
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]
    

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 32,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        max_len: int = 4096,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)

        center_idx = x.size(1) // 2
        x = self.norm(x[:, center_idx, :])

        return self.classifier(x)


# %%
#################  Function definitions  #################

#Load data
def loadData(path):  # load IQ data from CSV files in given folder path
    inputFiles = glob.glob(os.path.join(path, "*.csv"))
    if not inputFiles:
        raise FileNotFoundError(f"No CSV files found in: {path}")   
    
    dataFrame = pd.DataFrame()
    
    for file in inputFiles:
        df = pd.read_csv(file)
        dataFrame = pd.concat([dataFrame, df], ignore_index=True)
    label = dataFrame["label_index"].values
    return dataFrame, label

# Sliding windows
def make_windows(X: np.ndarray, y: np.ndarray, window: int, hop: int = 1):
    """Build sliding windows and use the center-symbol label as window target."""
    if len(X) < window:
        raise ValueError(f"Need at least {window} symbols, got {len(X)}")

    Xw, yw = [], []
    center = window // 2
    for start in range(0, len(X) - window + 1, hop):
    
        end = start + window
        Xw.append(X[start:end])
        yw.append(y[start + center])

    return np.stack(Xw).astype(np.float32), np.array(yw, dtype=np.int64)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# def count_parameters2(model):
#     compModel = torch.compile(model)
#     total_params = sum(p.numel() for p in compModel.parameters())
#     trainable_params = sum(p.numel() for p in compModel.parameters() if p.requires_grad)
#     return total_params, trainable_params

def get_model_size_mb(path):
    size_mb = os.path.getsize(path) / (1024 ** 2)
    return size_mb

###### Quantization function definitions #####

# def quantize_model(model, config):
#     quantized_model = copy.deepcopy(model)
#     quantize_(quantized_model, config)
#     return quantized_model


###### Pruning function definitions #####

def structured_prune_ffn(model, amount):
    for module in model.modules(): # go through the modules in the model
        if isinstance(module, nn.TransformerEncoderLayer): # if the current module is an instance of the TransformerEncoder class
            # Prune linear1 (expand layer)
            prune.ln_structured( # prune the linear layer "linear1"
                module.linear1, # the tensor to prune is the linear1 layer
                name="weight", # the parameter to prune based on
                amount=amount, # pruning ratio
                n=2, # the 2-norm is used.
                dim=0,  # prune neurons (rows)
            )

            # Prune linear2 (project back)
            prune.ln_structured( # prune the linear layer "linear2"
                module.linear2, # the tensor to prune is the linear2 layer
                name="weight", # the parameter to prune based on
                amount=amount, # pruning ratio
                n=2, # the 2-norm is used
                dim=1,  # match dimension  
            )
    return model

def rebuild_model_pruned(pruned_model, threshold=0.0):
    """
    Rebuild a pruned TransformerClassifier by shrinking FFN (feed-forward network) widths inside each
    TransformerEncoderLayer.

    This keeps d_model and attention dimensions unchanged (safe for MHA (multi-head attention)), and
    compacts only linear1/linear2 in each encoder layer.

    threshold: consider absolute weights <= threshold as pruned.
    Returns a new model with smaller FFN sub-layers.
    """
    #new_model = copy.deepcopy(pruned_model)
    new_model = pruned_model # simply a reference variable to pruned model
    kept_dims = []

    # raise an error if the given model does not have encoder layers
    if not hasattr(new_model, "encoder") or not hasattr(new_model.encoder, "layers"):
        raise ValueError("Expected model.encoder.layers for TransformerClassifier-like model")

    # loop through all the encoder layers
    for layer_idx, layer in enumerate(new_model.encoder.layers):
        linear1 = layer.linear1  # linear1 layer of current encoder layer
        linear2 = layer.linear2  # linear2 layer of current encoder layer

        # raise error if linear1 or linear2 are not part of this encoder layer
        if not isinstance(linear1, nn.Linear) or not isinstance(linear2, nn.Linear):
            raise ValueError(f"Encoder layer {layer_idx} does not have linear1/linear2 as nn.Linear")

       
        # tensor of booleans checking if weight values are greater than threshold (usually threshold = 0).
        # if any of these give "true", the method ".any()" turns the output into a single boolean saying "true".
        l1_row_active = (linear1.weight.data.abs() > threshold).any(dim=1)
        # if there is a bias value:
        # boolean expression whether bias is greater than threshold is bitwise OR'ed onto l1_row_active
        if linear1.bias is not None: 
            l1_row_active |= (linear1.bias.data.abs() > threshold)

        # same boolean expression principle as for l1_row_active, but for l2 columns (dim=0) this time
        l2_col_active = (linear2.weight.data.abs() > threshold).any(dim=0)
        
        # only keep FFN hidden units that are active in both projections by using bitwise AND.
        keep_mask = l1_row_active & l2_col_active  

        # Guarantee at least one hidden unit so the layer still works:
        # if there is no "true" at all in keep_mask,
        # locate index of the most important weight and set that index true in keep_mask
        if not keep_mask.any():
            keep_idx = torch.argmax(linear1.weight.data.abs().sum(dim=1)).item()
            keep_mask[keep_idx] = True

        # torch.where gets the indices where keepmask is true
        keep_indices = torch.where(keep_mask)[0]
        # the kept dimension of the layer is the number of elements in keep_indices
        kept_dim = int(keep_indices.numel()) 
        kept_dims.append(kept_dim)

        device = linear1.weight.device # device same as before
        dtype = linear1.weight.dtype # data type same as before

        # creating new linear layers to use for linear1 and linear2 with the reduced dimension kept_dim 
        new_linear1 = nn.Linear(
            linear1.in_features,
            kept_dim,
            bias=linear1.bias is not None,
            device=device,
            dtype=dtype,
        )
        new_linear2 = nn.Linear(
            kept_dim,
            linear2.out_features,
            bias=linear2.bias is not None,
            device=device,
            dtype=dtype,
        )
        # 
        with torch.no_grad(): # turns off gradient calculation
            # copy weights and biases from old linear1 according to the indices that we want to keep
            new_linear1.weight.copy_(linear1.weight.data[keep_indices, :])
            if linear1.bias is not None:
                new_linear1.bias.copy_(linear1.bias.data[keep_indices])
            
            # copy weights from old linear2 according to the indices that we want to keep. Copy all biases
            new_linear2.weight.copy_(linear2.weight.data[:, keep_indices])
            if linear2.bias is not None:
                new_linear2.bias.copy_(linear2.bias.data)

        layer.linear1 = new_linear1 # reassign linear1 layer of current encoder layer to the pruned version
        layer.linear2 = new_linear2 # reassign linear2 layer of current encoder layer to the pruned version

    print(f"Rebuilt Transformer FFN dims per encoder layer: {kept_dims}")
    return new_model


###### Knowledge Distilation function definitions #####
def kd_loss_fn(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.7):
    hard_loss = F.cross_entropy(student_logits, labels)

    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss

def evaluate_loss_acc(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * yb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_count += yb.size(0)

    return total_loss / total_count, total_correct / total_count


# KD Definition
def knowledgeDistilation(teacherModel,test_loader,train_loader, val_loader, device, lossTitle = "KD - Student Train and Validation loss", accTitle = "KD - Student Train and Validation accuracy", num_epochs = 100, learning_rate = 1e-3, weight_decay = 1e-4, student_config=None ):
    num_classes = 16
    
    if student_config is None:
        student_config = {
        "num_layers": 8,
        "d_model": 32,
        "nhead": 4,
        "dim_feedforward": 256,
    }
    
    student = TransformerClassifier(
    input_dim=2,
    num_classes=16,
    d_model=student_config["d_model"],
    nhead=student_config["nhead"],
    num_layers=student_config["num_layers"],
    dim_feedforward=student_config["dim_feedforward"],
    dropout=0.3,
    max_len=4096,
    ).to(device)
    
    temperature = 2
    alpha = 1
    
    # --------------------------------------------------
    # Teacher evaluation
    # --------------------------------------------------
    teacher_test_loss, teacher_test_acc = evaluate_loss_acc(teacherModel, test_loader, device)
    print(f"\nTeacher test loss: {teacher_test_loss:.4f} | Teacher test acc: {teacher_test_acc:.4f}")
    print("###Student Training###")
    # --------------------------------------------------
    # KD training
    # --------------------------------------------------
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    no_new_learning_counter = 0
    no_new_learning_threshold = 3
    lowest_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        student.train()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            with torch.no_grad():
                teacher_logits = teacherModel(xb)

            student_logits = student(xb)

            loss = kd_loss_fn(
                student_logits,
                teacher_logits,
                yb,
                temperature=temperature,
                alpha=alpha,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * yb.size(0)
            total_correct += (student_logits.argmax(dim=1) == yb).sum().item()
            total_count += yb.size(0)
        epoch_loss = total_loss / total_count
        if epoch_loss < lowest_loss:
            lowest_loss = epoch_loss
            no_new_learning_counter = 0
        else:
            no_new_learning_counter += 1
        if no_new_learning_counter >= no_new_learning_threshold:
            print(f"Early stopping at epoch {epoch} due to no improvement in loss for {no_new_learning_threshold} consecutive epochs.")
            break
    
        

    train_loss = total_loss / total_count
    train_acc = total_correct / total_count

    val_loss, val_acc = evaluate_loss_acc(student, val_loader, device)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(
    f"Epoch {epoch:03d}/{num_epochs} | "
    f"Train loss={train_loss:.4f} Train Accuracy={train_acc:.4f} | "
    f"Validation loss={val_loss:.4f} Validation Accuracy={val_acc:.4f}"
    )
    return student

    
###### Model evaluation definition #####

def totalEvaulation(model, test_loader, device,pipeLine = "Pipeline structure - Placeholder", CM_title = "Model Performance Evaluation - Placeholder",CM_SaveName = "CM Save Name Placeholder", fileName = "model_metrics_placeholder", show_cm=True):
    model.eval()
    num_classes = 16
    #path = f"{model.name()}.pth"
    torch.save(model.state_dict(), "modelSnapshot.pth")
    path = "modelSnapshot.pth"
    preds = []
    targets = []
    file_sizes = []
    accuracies = []
    total_times = []
    avg_times_per_sample = []
    avg_times_per_batch = []
    parameter_counts = []
    classification_reports = []
    torch.save(model.state_dict(), path)
    dataType = model.parameters().__next__().dtype
    bit_width = re.findall(r'[0-9]+', str(dataType))[0]
    
    
    
    intraDataFrame = pd.DataFrame({
    "Accuracy":[],
    "Precision" : [],
    "F1-score":[],
    "Total Inference time": [],
    "Batch Inference time":[],
    "Est. Sample Inference time": [],
    "Total parameters":[],
    "Memory footprint":[],
    "Data type":[],
    "Bit width":[]
    })
    

    n_samples = 0 # variable to hold how many samples through all batches
    batch_inference_times = []
    with torch.no_grad():
        for x_batch, y_batch, in test_loader:
            x_batch = x_batch.to(device)
            if device.type == "cuda": # synchronize cuda before measuring the time
                torch.cuda.synchronize()
            start_time = time.perf_counter()  # start time to measure inference (batch?)
            #outputs = prune_increment_model(x_batch)
            outputs = model(x_batch)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()
            if device.type == "cuda": # synchronize cuda before measuring the time
                torch.cuda.synchronize()
            end_time = time.perf_counter()  # end time to measure inference (batch?)
            preds.extend(predicted)
            targets.extend(y_batch.cpu())

            n_samples += x_batch.size(0)
            batch_inference_time = end_time - start_time
            batch_inference_times.append(batch_inference_time)
    
    report_dict = classification_report(targets, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(targets, preds)

    total_time = np.sum(batch_inference_times)
    avg_time_per_sample = total_time/n_samples
    avg_time_per_batch = np.mean(batch_inference_times)

    total_parameters, trainable_parameters = count_parameters(model)
    accuracy = accuracy_score(targets,preds) * 100
    size_mb = os.path.getsize(path) / (1024 ** 2)
    accuracies.append(accuracy) #calculate current accuracy
    total_times.append(total_time)
    avg_times_per_sample.append(avg_time_per_sample)
    avg_times_per_batch.append(avg_time_per_batch)
    parameter_counts.append(total_parameters)
    classification_reports.append(report_dict)

    if show_cm:
        fig, ax = plt.subplots(figsize=(10, 10))
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=np.arange(num_classes),
        ).plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            xticks_rotation=0,
        )

        ax.set_title(f"{CM_title}")
        plt.tight_layout()
        plt.savefig(f"{CM_SaveName}.png", dpi=1200, bbox_inches="tight")
        plt.show()



    Performance_metrics_line = pd.DataFrame({
    
    "Accuracy":[accuracy],
    "F1-score":[report_dict["macro avg"]["f1-score"]*100],
    "Precision":[report_dict["macro avg"]["precision"]*100],
    "Total Inference time": [total_times[-1]],
    "Batch Inference time":[avg_times_per_batch[-1]],
    "Est. Sample Inference time": [avg_times_per_sample[-1]],
    "Total parameters":[total_parameters],
    "Memory footprint":[size_mb],
    "Data type":[dataType],
    "Bit width":[bit_width]
    })
    Performance_metrics_dataframe = pd.concat([intraDataFrame,Performance_metrics_line])
    
    savedData = pickle.dump(Performance_metrics_dataframe, open(f"{fileName}.pkl", "wb"))
    return Performance_metrics_dataframe





# %%
#################################### Main code ########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = Path(r"/ceph/project/6G_inFactory/Model_Tests/Test_Pipeline/KD/model_16qam_rician_channel_snr_10.pth")
modelLoaded = torch.load(model_path, map_location=device, weights_only=False)
originalModel = copy.deepcopy(modelLoaded) 
modelForQuantization = copy.deepcopy(modelLoaded)
modelForPruning = copy.deepcopy(modelLoaded)
modelForKD = copy.deepcopy(modelLoaded)
modelForPipeline = copy.deepcopy(modelLoaded)



# %%
### Knowledge Distillation experiment definition #####

def kd_parameter_experiment(
    teacher_model,
    test_loader,
    train_loader,
    val_loader,
    device,
    param_name,
    param_values,
    output_csv
):
    results = []

    base_config = {
        "num_layers": 32,
        "d_model": 64,
        "nhead": 8,
        "dim_feedforward": 512,
    }

    for value in param_values:
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print("\n==============================")
        print(f"KD experiment: {param_name} = {value}")
        print("==============================")

        student_config = base_config.copy()
        student_config[param_name] = value

        if student_config["d_model"] % student_config["nhead"] != 0:
            print(
                f"Skipping {param_name}={value} because "
                f"d_model={student_config['d_model']} is not divisible by nhead={student_config['nhead']}"
            )
            continue

        # if student_config["d_model"] // student_config["nhead"] < 2:
        #     print(
        #         f"Skipping {param_name}={value} because "
        #         f"head_dim={student_config['d_model'] // student_config['nhead']} is too small"
        #     )
        #     continue

        student_model = knowledgeDistilation(
            teacher_model,
            test_loader,
            train_loader,
            val_loader,
            device,
            num_epochs=100,
            learning_rate=1e-3,
            weight_decay=1e-4,
            student_config=student_config,
        )

        metrics_df = totalEvaulation(
            student_model,
            test_loader,
            device,
            CM_SaveName=f"kd_{param_name}_{value}_cm",
            CM_title=f"KD Student - {param_name}={value}",
            fileName=f"kd_metrics_{param_name}_{value}",
            show_cm=False,
        )

        metrics_df["Experiment Parameter"] = param_name
        metrics_df["Parameter Value"] = value

        results.append(metrics_df)

        del student_model

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(output_csv, index=False)

    return results_df

# %%
### Experiment parameters and data loading ###
seq_len = 128
stride = 8
batch_size = 256

#data_dir = Path(r"C:\Users\mikep\Documents\GitHub\6G_inFactory\testPipeline\individualTesting\Michail\evaluationDatasetMean5")
#data_dir = Path(r"C:\Users\mikep\Documents\GitHub\6G_inFactory\2. testingWithRayleigh")
#data_dir = Path(r"C:\GitHub\6G_inFactory\2. testingWithRayleigh")
#data_dir = Path(r"C:\GitHub Clones\6G_inFactory\2. testingWithRayleigh")
# data_dir = Path(r"C:\Users\emils\Documents\GitHub\6G_inFactory\2. testingWithRayleigh")
# dataPath = sorted(data_dir.glob("4qam_noisy_with_labels_rician_fading_k_2.16_evaluation_dataset1.csv"))

csv_path = Path(r"/ceph/project/6G_inFactory/Model_Tests/Test_Pipeline/KD/16qam_noisy_with_labels_rician_fading_k_2.16_snr_10_evaluation_dataset2.csv")

# The correct loading of data

# Load synthetic QAM symbol-level data
dfs = [] # dataframes

# for file in dataPath:
#     df = pd.read_csv(file)
#     dfs.append(df)
# df_all = pd.concat(dfs, ignore_index=True)

df_all = pd.read_csv(csv_path)

required_cols = ["noisy_I", "noisy_Q", "clean_I", "clean_Q", "label_index"]
missing = [c for c in required_cols if c not in df_all.columns]
if missing:
    raise ValueError(f"Missing columns in {csv_path}: {missing}")

# Labels
y_symbol = df_all["label_index"].to_numpy(dtype=np.int64)

# Clean input for training
#X_clean = df_all[["clean_I", "clean_Q"]].to_numpy(dtype=np.float32)

# Noisy input for validation/testing
X_noisy = df_all[["noisy_I", "noisy_Q"]].to_numpy(dtype=np.float32)

num_classes = int(y_symbol.max()) + 1
print(f"Loaded {len(df_all)} symbols | classes: {num_classes}")


# the correct splitting of data

# Split on symbol stream first to avoid overlap leakage across splits
n_total = len(y_symbol)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)

# Train on noisy
X_train_sym = X_noisy[0:n_train]
y_train_sym = y_symbol[0:n_train]

# Validate on noisy
X_val_sym = X_noisy[n_train : n_train + n_val]
y_val_sym = y_symbol[n_train : n_train + n_val]

# Test on noisy
X_test_sym = X_noisy[n_train + n_val:]
y_test_sym = y_symbol[n_train + n_val:]

X_train, y_train = make_windows(X_train_sym, y_train_sym, seq_len, stride)
X_val, y_val = make_windows(X_val_sym, y_val_sym, seq_len, stride)
X_test, y_test = make_windows(X_test_sym, y_test_sym, seq_len, stride)

print("Windowed shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape, y_val.shape)
print("Test: ", X_test.shape, y_test.shape)



# %%
# the correct dataloader creation

# Training - Validation - Test Datasets
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

print(f"Train/Val/Test windows: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")



# %%
### Evaluate the original model before any modifications
original_metrics = totalEvaulation(
    originalModel,
    test_loader,
    device,
    CM_SaveName="original_model_cm",
    CM_title="Original Model Performance",
    fileName="original_model_metrics",
    show_cm=False
)

# %%
### Plotting function definition for KD experiment results ###
def plot_kd_metric(
    df,
    x_col,
    metric_name,
    ylabel,
    save_name,
    parameter_name,
    baseline_value=None
):
    plt.figure(figsize=(15, 8))

    x_values = df[x_col].astype(str)
    y_values = df[metric_name]
    param_values = df["Total parameters"]
    baselineParams = original_metrics["Total parameters"][0]
    siBaseValue = si_format(baselineParams,precision=3)
    bars = []
    #height = []

    ax = plt.gca()
    for i in range(len(df)):
        shortParam = param_values.iloc[i]
        shortParamVal = si_format(shortParam,precision=3)
        #label = f"{x_values.iloc[i]} → {param_values.iloc[i]:,} params"
        label = f"{x_values.iloc[i]} - {shortParamVal} params"

        bar_container = plt.bar(
            x_values.iloc[i],
            y_values.iloc[i],
            edgecolor="black",
            label=label
        )

        bars.extend(bar_container)


    if baseline_value is not None:
        
        if metric_name in ["Memory footprint"]:
                baseMetric = baseline_value
                sizeUnit = 'MB'
                ax.yaxis.set_major_formatter(mticker.EngFormatter(useMathText=True,unit=sizeUnit))
                baseSI = si_format(baseMetric,precision=4)
                x = baseline_value/np.mean(df[metric_name])*0.1
                y = baseline_value/np.mean(df["Parameter Value"][1])*4.05
                ax.text(x,y,f"Original Model - {baseSI}MB",color='red', fontsize=12, fontweight="bold")
                for bar in bars:
                    heightMem = bar.get_height()
                    siHeightMem = si_format(heightMem,precision=3)
                    plt.annotate(
                        f"{siHeightMem} MB",
                        xy=(bar.get_x() + bar.get_width() / 2, heightMem),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        fontweight="bold"
                    )

                plt.axhline(
                y=baseline_value,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Original Model \n({siBaseValue} params)"
                )
                plt.text(
                x=len(x_values) - 1,                 # right side of plot
                y=baseline_value,
                #s=f"{baseline_value:.2f}",
                s="",
                color="red",
                fontsize=12,
                fontweight="bold",
                ha="right",
                va="bottom"
                )
                
        if metric_name in ["Est. Sample Inference time", "Batch Inference time","Total Inference time"]:
                ax = plt.gca()
                originalMetric = baseline_value
                timeUnit = 's'
                ax.yaxis.set_major_formatter(mticker.EngFormatter(useMathText=True,unit=timeUnit))
                siTest = si_format(originalMetric,precision=3)
                x = baseline_value/np.mean(df[metric_name])*0.1
                y = baseline_value/np.mean(df["Parameter Value"][1])*4.05
                ax.text(x,y,f"Original Model - {siTest}s",color='red', fontsize=12, fontweight="bold")

                for bar in bars:
                    heightTime = bar.get_height()
                    siHeightTime = si_format(heightTime,precision=3)
                    plt.annotate(
                        f"{siHeightTime}s",
                        xy=(bar.get_x() + bar.get_width() / 2, heightTime),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        fontweight="bold"
                    )



                plt.axhline(
                y=baseline_value,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Original Model \n({siBaseValue} params)"
            )
                
                plt.text(
                x=len(x_values) - 1,                 # right side of plot
                y=baseline_value,
                #s=f"{baseline_value:.2f}",
                s="",
                color="red",
                fontsize=12,
                fontweight="bold",
                ha="right",
                va="bottom"
            )
                
        if metric_name in ["Accuracy", "Precision", "F1-score"]:
            plt.ylim(0, 110)
            ax = plt.gca()
            originalMetric = baseline_value
            siTest = si_format(originalMetric,precision=3)
            x = baseline_value/np.mean(df[metric_name])*0.1
            x = plt.xlim()[1]*0.3
            y = plt.ylim()[1]*0.95
            ax.text(x,y,f"Original Model - {siTest}%",color='red', fontsize=12, fontweight="bold")
            for bar in bars:
                height = bar.get_height()


                plt.annotate(
                f"{height:.3f} %",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold"
            )
            
            plt.axhline(
                y=baseline_value,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Original Model \n({siBaseValue} params)"
            )

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=14,
    )
    
    if metric_name in ["Accuracy", "Precision", "F1-score"]:
        plt.ylim(0, 110)

    if metric_name in["Est. Sample Inference time", "Batch Inference time","Total Inference time"]:
        ax = plt.gca()
        #timeUnit = "s"
        #ax.yaxis.set_major_formatter(mticker.EngFormatter(useOffset=False,unit=timeUnit))
        #mticker.ScalarFormatter

    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.yticks(fontsize=14)
    plt.xticks(rotation=0, fontsize=14)
    plt.xlabel(parameter_name, fontsize=16, fontweight="bold")
    plt.ylabel(ylabel, fontsize=16, fontweight="bold")
    plt.title(f"{ylabel} vs {parameter_name}", fontsize=20, fontweight="bold")

    plt.tight_layout()

    plt.savefig(save_name, dpi=300, bbox_inches="tight")

    plt.show()

# %%
### Applying KD Only #####
studentModel = knowledgeDistilation(originalModel,test_loader,train_loader,val_loader,device, lossTitle="KD - Student Train and Validation Loss in KD Only", accTitle="KD - Student Train and Validation Accuracy in KD Only")
totalEvaulation(studentModel, test_loader, device, CM_SaveName="kd_model_cm",CM_title = "KD Model Performance Evaluation", fileName = "kd_model_metrics")
num_layers = [2,4,8,16]
#num_layers = [2,4,8]
d_model = [8,16,32]
n_head = [1,2,4]
ffn = [2,4,8,16,32,64,128,256]



# %%
### Num Layers Experiment

kd_num_layers_df = kd_parameter_experiment(
    originalModel,
    test_loader,
    train_loader,
    val_loader,
    device,
    "num_layers",
    num_layers,
    "kd_num_layers_metrics.csv"
)


# %%
kd_num_layers_df = pd.read_csv("kd_num_layers_metrics.csv")
original_metrics = pd.read_pickle("original_model_metrics.pkl")

plot_kd_metric(kd_num_layers_df, 
               "Parameter Value", 
               "Accuracy", "Accuracy (%)", 
               "kd_accuracy_vs_num_layers.png", 
               "# Layers", 
               baseline_value=original_metrics["Accuracy"].iloc[0])
plot_kd_metric(kd_num_layers_df, 
               "Parameter Value", 
               "Precision", 
               "Precision (%)", 
               "kd_precision_vs_num_layers.png", 
               "# Layers", baseline_value=original_metrics["Precision"].iloc[0])
plot_kd_metric(kd_num_layers_df, 
               "Parameter Value", 
               "F1-score", 
               "F1-score (%)", 
               "kd_f1_vs_num_layers.png", 
               "# Layers", baseline_value=original_metrics["F1-score"].iloc[0])
plot_kd_metric(kd_num_layers_df, 
               "Parameter Value", 
               "Memory footprint", 
               "Memory Footprint", 
               "kd_memory_vs_num_layers.png", 
               "# Layers", baseline_value=original_metrics["Memory footprint"].iloc[0])
#plot_kd_metric(kd_num_layers_df, "Parameter Value", "Total parameters", "Total Parameters", "kd_params_vs_num_layers.png", "# Layers", baseline_value=original_metrics["Total parameters"].iloc[0])
plot_kd_metric(kd_num_layers_df, 
               "Parameter Value", 
               "Est. Sample Inference time", 
               "Estimated Sample Inference Time", 
               "kd_inference_vs_num_layers.png", 
               "# Layers", baseline_value=original_metrics["Est. Sample Inference time"].iloc[0])
plot_kd_metric(kd_num_layers_df, 
               "Parameter Value", 
               "Batch Inference time", 
               "Batch Inference time", 
               "kd_batchinference_vs_num_layers.png", 
               "# Layers", baseline_value=original_metrics["Batch Inference time"].iloc[0])
plot_kd_metric(kd_num_layers_df, 
               "Parameter Value", 
               "Total Inference time", 
               "Total Inference time", 
               "kd_totalinference_vs_num_layers.png", 
               "# Layers", baseline_value=original_metrics["Total Inference time"].iloc[0])


# %%
kd_num_dmodel_df = kd_parameter_experiment(
    originalModel,
    test_loader,
    train_loader,
    val_loader,
    device,
    "d_model",
    d_model,
    "kd_d_model_metrics.csv"
)

# %%
plot_kd_metric(kd_num_dmodel_df, "Parameter Value", "Accuracy", "Accuracy (%)", "kd_accuracy_vs_d_model.png", "# Embedding Dimension", baseline_value=original_metrics["Accuracy"].iloc[0])
plot_kd_metric(kd_num_dmodel_df, "Parameter Value", "Precision", "Precision (%)", "kd_precision_vs_d_model.png", "# Embedding Dimension", baseline_value=original_metrics["Precision"].iloc[0])
plot_kd_metric(kd_num_dmodel_df, "Parameter Value", "F1-score", "F1-score (%)", "kd_f1_vs_d_model.png", "# Embedding Dimension", baseline_value=original_metrics["F1-score"].iloc[0])
plot_kd_metric(kd_num_dmodel_df, "Parameter Value", "Memory footprint", "Memory Footprint (MB)", "kd_memory_vs_d_model.png", "# Embedding Dimension", baseline_value=original_metrics["Memory footprint"].iloc[0])
#plot_kd_metric(kd_num_dmodel_df, "Parameter Value", "Total parameters", "Total Parameters", "kd_params_vs_d_model.png", "# Embedding Dimension", baseline_value=original_metrics["Total parameters"].iloc[0])
plot_kd_metric(kd_num_dmodel_df, "Parameter Value", "Est. Sample Inference time", "Estimated Sample Inference Time (ms)", "kd_inference_vs_d_model.png", "# Embedding Dimension", baseline_value=original_metrics["Est. Sample Inference time"].iloc[0])
plot_kd_metric(kd_num_dmodel_df, "Parameter Value", "Batch Inference time", "Batch Inference time (ms)", "kd_batchinference_vs_d_model.png", "# Embedding Dimension", baseline_value=original_metrics["Batch Inference time"].iloc[0])
plot_kd_metric(kd_num_dmodel_df, "Parameter Value", "Total Inference time", "Total Inference time (s)", "kd_totalinference_vs_d_model.png", "# Embedding Dimension", baseline_value=original_metrics["Total Inference time"].iloc[0])

# %%
kd_num_nhead_df = kd_parameter_experiment(
    originalModel,
    test_loader,
    train_loader,
    val_loader,
    device,
    "nhead",
    n_head,
    "kd_n_head_metrics.csv"
)

# %%
plot_kd_metric(kd_num_nhead_df, 
               "Parameter Value", 
               "Accuracy", 
               "Accuracy (%)", 
               "kd_accuracy_vs_n_head.png", 
               "# Heads", baseline_value=original_metrics["Accuracy"].iloc[0])
plot_kd_metric(kd_num_nhead_df, 
               "Parameter Value", 
               "Precision", 
               "Precision (%)", 
               "kd_precision_vs_n_head.png", 
               "# Heads", baseline_value=original_metrics["Precision"].iloc[0])
plot_kd_metric(kd_num_nhead_df, 
               "Parameter Value", 
               "F1-score", 
               "F1-score (%)", 
               "kd_f1_vs_n_head.png", 
               "# Heads", baseline_value=original_metrics["F1-score"].iloc[0])
plot_kd_metric(kd_num_nhead_df, 
               "Parameter Value", 
               "Memory footprint", 
               "Memory Footprint (MB)", 
               "kd_memory_vs_n_head.png", 
               "# Heads", baseline_value=original_metrics["Memory footprint"].iloc[0])
#plot_kd_metric(kd_num_nhead_df, "Parameter Value", "Total parameters", "Total Parameters", "kd_params_vs_n_head.png", "# Heads", baseline_value=original_metrics["Total parameters"].iloc[0])
plot_kd_metric(kd_num_nhead_df, 
               "Parameter Value", 
               "Est. Sample Inference time", 
               "Estimated Sample Inference Time (ms)", 
               "kd_inference_vs_n_head.png", 
               "# Heads", baseline_value=original_metrics["Est. Sample Inference time"].iloc[0])
plot_kd_metric(kd_num_nhead_df, 
               "Parameter Value", "Batch Inference time", 
               "Batch Inference time (ms)", 
               "kd_batchinference_vs_n_head.png", 
               "# Heads", baseline_value=original_metrics["Batch Inference time"].iloc[0])
plot_kd_metric(kd_num_nhead_df, 
               "Parameter Value", 
               "Total Inference time", 
               "Total Inference time (s)", 
               "kd_totalinference_vs_n_head.png", 
               "# Heads", baseline_value=original_metrics["Total Inference time"].iloc[0])


# %%
kd_num_ffn_df = kd_parameter_experiment(
    originalModel,
    test_loader,
    train_loader,
    val_loader,
    device,
    "dim_feedforward",
    ffn,
    "kd_ffn_metrics.csv"
)

# %%
plot_kd_metric(kd_num_ffn_df,
                "Parameter Value", 
                "Accuracy", 
                "Accuracy (%)", 
                "kd_accuracy_vs_ffn.png", 
                "# Hidden Dimensions", 
                baseline_value=original_metrics["Accuracy"].iloc[0]
                )
plot_kd_metric(kd_num_ffn_df, 
               "Parameter Value", 
               "Precision", 
               "Precision (%)", 
               "kd_precision_vs_ffn.png", 
               "# Hidden Dimensions", 
               baseline_value=original_metrics["Precision"].iloc[0]
               )
plot_kd_metric(kd_num_ffn_df, 
               "Parameter Value", 
               "F1-score", 
               "F1-score (%)", 
               "kd_f1_vs_ffn.png", 
               "# Hidden Dimensions", 
               baseline_value=original_metrics["F1-score"].iloc[0]
               )
plot_kd_metric(kd_num_ffn_df, 
               "Parameter Value", 
               "Memory footprint", 
               "Memory Footprint (MB)", 
               "kd_memory_vs_ffn.png", 
               "# Hidden Dimensions", 
               baseline_value=original_metrics["Memory footprint"].iloc[0]
               )
#plot_kd_metric(kd_num_ffn_df, "Parameter Value", "Total parameters", "Total Parameters", "kd_params_vs_ffn.png", "# Hidden Dimensions", baseline_value=original_metrics["Total parameters"].iloc[0])
plot_kd_metric(kd_num_ffn_df, 
               "Parameter Value", 
               "Est. Sample Inference time", 
               "Estimated Sample Inference Time (ms)", 
               "kd_inference_vs_ffn.png", 
               "# Hidden Dimensions", 
               baseline_value=original_metrics["Est. Sample Inference time"].iloc[0]
               )
plot_kd_metric(kd_num_ffn_df, 
               "Parameter Value", 
               "Batch Inference time", 
               "Batch Inference time (ms)", 
               "kd_batchinference_vs_ffn.png", 
               "# Hidden Dimensions", 
               baseline_value=original_metrics["Batch Inference time"].iloc[0]
               )
plot_kd_metric(kd_num_ffn_df, 
               "Parameter Value", 
               "Total Inference time", 
               "Total Inference time (s)", 
               "kd_totalinference_vs_ffn.png", 
               "# Hidden Dimensions", 
               baseline_value=original_metrics["Total Inference time"].iloc[0]
               )


