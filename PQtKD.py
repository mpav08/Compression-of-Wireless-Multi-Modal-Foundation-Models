# =========================================================
# Clean Prune -> Quantize -> KD pipeline
# =========================================================

import copy
import math
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# Model definitions
# =========================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
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
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
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

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        center_idx = x.size(1) // 2
        x = self.norm(x[:, center_idx, :])
        return self.classifier(x)


# =========================================================
# Data utilities
# =========================================================

def make_windows(X: np.ndarray, y: np.ndarray, window: int, hop: int = 1):
    if len(X) < window:
        raise ValueError(f"Need at least {window} symbols, got {len(X)}")

    Xw, yw = [], []
    center = window // 2

    for start in range(0, len(X) - window + 1, hop):
        end = start + window
        Xw.append(X[start:end])
        yw.append(y[start + center])

    return np.stack(Xw).astype(np.float32), np.array(yw, dtype=np.int64)


def build_dataloaders(csv_path, seq_len=128, stride=8, batch_size=256):
    df = pd.read_csv(csv_path)

    required_cols = ["noisy_I", "noisy_Q", "clean_I", "clean_Q", "label_index"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    y_symbol = df["label_index"].to_numpy(dtype=np.int64)
    X_noisy = df[["noisy_I", "noisy_Q"]].to_numpy(dtype=np.float32)
    num_classes = int(y_symbol.max()) + 1

    n_total = len(y_symbol)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    X_train_sym = X_noisy[:n_train]
    y_train_sym = y_symbol[:n_train]

    X_val_sym = X_noisy[n_train:n_train + n_val]
    y_val_sym = y_symbol[n_train:n_train + n_val]

    X_test_sym = X_noisy[n_train + n_val:]
    y_test_sym = y_symbol[n_train + n_val:]

    X_train, y_train = make_windows(X_train_sym, y_train_sym, seq_len, stride)
    X_val, y_val = make_windows(X_val_sym, y_val_sym, seq_len, stride)
    X_test, y_test = make_windows(X_test_sym, y_test_sym, seq_len, stride)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Loaded {len(df)} symbols | classes: {num_classes}")
    print(f"Train/Val/Test windows: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    return train_loader, val_loader, test_loader, num_classes


# =========================================================
# Metrics utilities
# =========================================================

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)


# =========================================================
# Pruning
# =========================================================

def compute_gradient_importance_scores(model, calibration_loader, device, loss_fn=None):
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    model.eval()
    importance_scores = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            importance_scores[name] = torch.zeros_like(param, device="cpu", dtype=torch.float32)

    num_batches = 0

    for xb, yb in calibration_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        model.zero_grad(set_to_none=True)
        outputs = model(xb)
        loss = loss_fn(outputs, yb)
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue

                score = (param.detach() * param.grad.detach()).abs().to("cpu", dtype=torch.float32)
                importance_scores[name] += score

        num_batches += 1

    if num_batches == 0:
        raise ValueError("calibration_loader produced no batches")

    for name in importance_scores:
        importance_scores[name] /= num_batches

    return importance_scores


def structured_prune_ffn(model, amount, importance_scores):
    layer_count = 0

    for module in model.modules():
        if isinstance(module, nn.TransformerEncoderLayer):
            key_l1 = f"encoder.layers.{layer_count}.linear1.weight"
            key_l2 = f"encoder.layers.{layer_count}.linear2.weight"

            prune.ln_structured(
                module.linear1,
                name="weight",
                amount=amount,
                n=2,
                dim=0,
                importance_scores=importance_scores[key_l1].to(module.linear1.weight.device),
            )

            prune.ln_structured(
                module.linear2,
                name="weight",
                amount=amount,
                n=2,
                dim=1,
                importance_scores=importance_scores[key_l2].to(module.linear2.weight.device),
            )

            layer_count += 1

    return model


def rebuild_model_pruned(pruned_model, threshold=0.0):
    new_model = pruned_model
    kept_dims = []

    if not hasattr(new_model, "encoder") or not hasattr(new_model.encoder, "layers"):
        raise ValueError("Expected model.encoder.layers for TransformerClassifier-like model")

    for layer_idx, layer in enumerate(new_model.encoder.layers):
        linear1 = layer.linear1
        linear2 = layer.linear2

        if not isinstance(linear1, nn.Linear) or not isinstance(linear2, nn.Linear):
            raise ValueError(f"Encoder layer {layer_idx} does not have linear1/linear2 as nn.Linear")

        l1_row_active = (linear1.weight.data.abs() > threshold).any(dim=1)
        if linear1.bias is not None:
            l1_row_active |= (linear1.bias.data.abs() > threshold)

        l2_col_active = (linear2.weight.data.abs() > threshold).any(dim=0)
        keep_mask = l1_row_active & l2_col_active

        if not keep_mask.any():
            keep_idx = torch.argmax(linear1.weight.data.abs().sum(dim=1)).item()
            keep_mask[keep_idx] = True

        keep_indices = torch.where(keep_mask)[0]
        kept_dim = int(keep_indices.numel())
        kept_dims.append(kept_dim)

        device = linear1.weight.device
        dtype = linear1.weight.dtype

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

        with torch.no_grad():
            new_linear1.weight.copy_(linear1.weight.data[keep_indices, :])
            if linear1.bias is not None:
                new_linear1.bias.copy_(linear1.bias.data[keep_indices])

            new_linear2.weight.copy_(linear2.weight.data[:, keep_indices])
            if linear2.bias is not None:
                new_linear2.bias.copy_(linear2.bias.data)

        layer.linear1 = new_linear1
        layer.linear2 = new_linear2

    print(f"Rebuilt Transformer FFN dims per encoder layer: {kept_dims}")
    return new_model


# =========================================================
# Manual quantization
# =========================================================

def absmax_quantize_tensor(tensor, bitwidth):
    if bitwidth == 1:
        qmin = 0
        qmax = 1
        storage_dtype = torch.int8
    elif bitwidth in [2, 4, 8]:
        qmin = -(2 ** (bitwidth - 1)) + 1
        qmax = (2 ** (bitwidth - 1)) - 1
        storage_dtype = torch.int8
    elif bitwidth == 16:
        qmin = -(2 ** 15)
        qmax = (2 ** 15) - 1
        storage_dtype = torch.int16
    elif bitwidth == 32:
        qmin = -(2 ** 31)
        qmax = (2 ** 31) - 1
        storage_dtype = torch.int32
    else:
        raise ValueError(f"Unsupported bitwidth: {bitwidth}")

    max_abs = tensor.abs().max()

    if max_abs == 0:
        S = torch.tensor(1.0, device=tensor.device)
        x_q = torch.zeros_like(tensor, dtype=storage_dtype)
        return x_q, S

    S = qmax / max_abs
    x_q = torch.round(tensor * S)
    x_q = torch.clamp(x_q, qmin, qmax)
    x_q = x_q.to(storage_dtype)

    return x_q, S


def dequantize_tensor(x_q, S):
    return x_q.float() / S


def pack_lowbit_signed(q_tensor, bitwidth):
    if bitwidth == 1:
        qmin = 0
    else:
        qmin = -(2 ** (bitwidth - 1))

    q_unsigned = (q_tensor.flatten().to(torch.int16) + abs(qmin)).to(torch.uint8)
    values_per_byte = 8 // bitwidth
    pad_len = (-q_unsigned.numel()) % values_per_byte

    if pad_len > 0:
        q_unsigned = torch.cat([q_unsigned, torch.zeros(pad_len, dtype=torch.uint8)])

    q_unsigned = q_unsigned.view(-1, values_per_byte)
    packed = torch.zeros(q_unsigned.size(0), dtype=torch.uint8)

    for i in range(values_per_byte):
        packed |= q_unsigned[:, i] << (i * bitwidth)

    return packed, pad_len


def unpack_lowbit_signed(packed, original_shape, bitwidth, pad_len):
    if bitwidth == 1:
        qmin = 0
    else:
        qmin = -(2 ** (bitwidth - 1))

    values_per_byte = 8 // bitwidth
    mask = (1 << bitwidth) - 1
    unpacked = []

    for i in range(values_per_byte):
        values = (packed >> (i * bitwidth)) & mask
        unpacked.append(values)

    q_unsigned = torch.stack(unpacked, dim=1).flatten()

    if pad_len > 0:
        q_unsigned = q_unsigned[:-pad_len]

    q_signed = q_unsigned.to(torch.int8) + qmin
    return q_signed.view(original_shape)


def save_quantized_model(model, path, bitwidth):
    quantized_state = {}

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "bias" in name or "norm" in name.lower():
                quantized_state[name] = {
                    "tensor": param.detach().cpu(),
                    "quantized": False,
                }
                continue

            x_q, S = absmax_quantize_tensor(param.detach().cpu(), bitwidth)

            if bitwidth in [1, 2, 4]:
                packed, pad_len = pack_lowbit_signed(x_q, bitwidth)

                quantized_state[name] = {
                    "tensor": packed,
                    "shape": tuple(x_q.shape),
                    "pad_len": pad_len,
                    "packed": True,
                    "quantized": True,
                    "bitwidth": bitwidth,
                    "S": S.cpu(),
                }
            else:
                quantized_state[name] = {
                    "tensor": x_q.cpu(),
                    "shape": tuple(x_q.shape),
                    "pad_len": 0,
                    "packed": False,
                    "quantized": True,
                    "bitwidth": bitwidth,
                    "S": S.cpu(),
                }

    torch.save(quantized_state, path)


def load_quantized_model(base_model, path, device):
    model = copy.deepcopy(base_model)
    quantized_state = torch.load(path, map_location="cpu")

    with torch.no_grad():
        for name, param in model.named_parameters():
            item = quantized_state[name]

            if not item["quantized"]:
                param.copy_(item["tensor"].to(param.device))
                continue

            bitwidth = item["bitwidth"]

            if item["packed"]:
                x_q = unpack_lowbit_signed(
                    item["tensor"],
                    item["shape"],
                    bitwidth,
                    item["pad_len"],
                )
            else:
                x_q = item["tensor"]

            S = item["S"]
            x_dequant = dequantize_tensor(x_q, S)
            param.copy_(x_dequant.to(param.device))

    model.to(device)
    model.eval()
    return model


# =========================================================
# Knowledge distillation
# =========================================================

def kd_loss_fn(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.7):
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


def knowledgeDistilation(
    teacherModel,
    test_loader,
    train_loader,
    val_loader,
    device,
    num_epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-4,
    student_config=None,
    num_classes=16,
):
    if student_config is None:
        student_config = {
            "num_layers": 2,
            "d_model": 8,
            "nhead": 4,
            "dim_feedforward": 2,
        }

    student = TransformerClassifier(
        input_dim=2,
        num_classes=num_classes,
        d_model=student_config["d_model"],
        nhead=student_config["nhead"],
        num_layers=student_config["num_layers"],
        dim_feedforward=student_config["dim_feedforward"],
        dropout=0.3,
        max_len=4096,
    ).to(device)

    temperature = 2.0
    alpha = 0.7

    teacher_test_loss, teacher_test_acc = evaluate_loss_acc(teacherModel, test_loader, device)
    print(f"Teacher test loss: {teacher_test_loss:.4f} | Teacher test acc: {teacher_test_acc:.4f}")

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

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

        train_loss = total_loss / total_count
        train_acc = total_correct / total_count
        val_loss, val_acc = evaluate_loss_acc(student, val_loader, device)

        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"Train loss={train_loss:.4f} Train acc={train_acc:.4f} | "
            f"Val loss={val_loss:.4f} Val acc={val_acc:.4f}"
        )

        if train_loss < lowest_loss:
            lowest_loss = train_loss
            no_new_learning_counter = 0
        else:
            no_new_learning_counter += 1

        if no_new_learning_counter >= no_new_learning_threshold:
            print(f"Early stopping at epoch {epoch}")
            break

    student.eval()
    return student


# =========================================================
# Evaluation
# =========================================================

def totalEvaulation(
    model,
    test_loader,
    device,
    CM_title="Model Performance Evaluation",
    CM_SaveName="confusion_matrix",
    fileName="model_metrics",
    show_cm=False,
):
    model.eval()
    path = "modelSnapshot.pth"
    torch.save(model.state_dict(), path)

    preds = []
    targets = []
    batch_inference_times = []
    n_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            outputs = model(x_batch)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            preds.extend(predicted)
            targets.extend(y_batch.cpu().numpy())
            n_samples += x_batch.size(0)
            batch_inference_times.append(end_time - start_time)

    report_dict = classification_report(targets, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(targets, preds)

    total_time = float(np.sum(batch_inference_times))
    avg_time_per_sample = total_time / n_samples
    avg_time_per_batch = float(np.mean(batch_inference_times))
    total_parameters, trainable_parameters = count_parameters(model)
    accuracy = accuracy_score(targets, preds) * 100
    size_mb = os.path.getsize(path) / (1024 ** 2)
    data_type = next(model.parameters()).dtype
    bit_width = re.findall(r"[0-9]+", str(data_type))[0]

    if show_cm:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=np.arange(16),
        ).plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            xticks_rotation=0,
        )
        ax.set_title(CM_title)
        plt.tight_layout()
        plt.savefig(f"{CM_SaveName}.png", dpi=1200, bbox_inches="tight")
        plt.show()

    metrics_df = pd.DataFrame({
        "Accuracy": [accuracy],
        "F1-score": [report_dict["macro avg"]["f1-score"] * 100],
        "Precision": [report_dict["macro avg"]["precision"] * 100],
        "Total Inference time": [total_time],
        "Batch Inference time": [avg_time_per_batch],
        "Est. Sample Inference time": [avg_time_per_sample],
        "Total parameters": [total_parameters],
        "Trainable parameters": [trainable_parameters],
        "Memory footprint": [size_mb],
        "Data type": [data_type],
        "Bit width": [bit_width],
    })

    with open(f"{fileName}.pkl", "wb") as f:
        pickle.dump(metrics_df, f)

    return metrics_df


# =========================================================
# Final pipeline: Prune -> Quantize -> KD
# =========================================================

def prune_quantize_kd(
    model,
    bitwidths,
    device,
    train_loader,
    test_loader,
    val_loader,
    pruning_ratio=0.25,
    output_csv="prune_quantize_kd_results.csv",
):
    results = []

    print("Computing pruning importance scores...")
    teacher_model = copy.deepcopy(model)

    importance_scores = compute_gradient_importance_scores(
        teacher_model,
        val_loader,
        device,
    )

    print("Pruning teacher...")
    pruned_teacher = structured_prune_ffn(
        teacher_model,
        amount=pruning_ratio,
        importance_scores=importance_scores,
    )

    rebuilt_teacher = rebuild_model_pruned(
        pruned_teacher,
        threshold=0.0,
    )

    for bitwidth in bitwidths:
        print("\n==============================")
        print(f"Prune -> Quantize -> KD | INT{bitwidth}")
        print("==============================")

        quant_teacher_path = f"prune_quant_teacher_int{bitwidth}.pth"

        save_quantized_model(
            rebuilt_teacher,
            quant_teacher_path,
            bitwidth,
        )

        quantized_teacher = load_quantized_model(
            rebuilt_teacher,
            quant_teacher_path,
            device,
        )

        kd_student = knowledgeDistilation(
            teacherModel=quantized_teacher,
            test_loader=test_loader,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        float_student_path = f"prune_quant_kd_student_int{bitwidth}_float32.pth"
        quant_student_path = f"prune_quant_kd_student_int{bitwidth}_quantized.pth"

        torch.save(kd_student, float_student_path)
        save_quantized_model(kd_student, quant_student_path, bitwidth)

        metrics_df = totalEvaulation(
            kd_student,
            test_loader,
            device,
            CM_title=f"Prune -> Quantize -> KD INT{bitwidth}",
            CM_SaveName=f"prune_quant_kd_cm_int{bitwidth}",
            fileName=f"prune_quant_kd_metrics_int{bitwidth}",
            show_cm=False,
        )

        metrics_df["Quantization Type"] = f"INT{bitwidth}"
        metrics_df["Bit width"] = bitwidth
        metrics_df["Pruning Ratio"] = pruning_ratio
        metrics_df["Teacher Quantized Size (MB)"] = get_model_size_mb(quant_teacher_path)
        metrics_df["Float32 Student Size (MB)"] = get_model_size_mb(float_student_path)
        metrics_df["Stored Model Size (MB)"] = get_model_size_mb(quant_student_path)

        results.append(metrics_df)

    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    return final_df


# =========================================================
# Run section
# =========================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(r"C:\Users\mikep\Documents\GitHub\Compression-of-Wireless-Multi-Modal-Foundation-Models\model_16qam_rician_channel_snr_5_epoch_11.pth")

    csv_path = Path(r"C:\Users\mikep\Documents\GitHub\Compression-of-Wireless-Multi-Modal-Foundation-Models\16qam_noisy_with_labels_rician_fading_k_2.16_snr_5_evaluation_dataset3.csv")

    originalModel = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    train_loader, val_loader, test_loader, num_classes = build_dataloaders(
        csv_path=csv_path,
        seq_len=128,
        stride=8,
        batch_size=256,
    )

    original_metrics = totalEvaulation(
        originalModel,
        test_loader,
        device,
        CM_SaveName="original_model_cm",
        CM_title="Original Model Performance",
        fileName="original_model_metrics",
        show_cm=False,
    )

    bitwidths = [1, 2, 4, 8, 16, 32]

    pqkd_results_df = prune_quantize_kd(
        model=originalModel,
        bitwidths=bitwidths,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        pruning_ratio=0.25,
        output_csv="prune_quantize_kd_results.csv",
    )

    print("\nFinal Prune -> Quantize -> KD results:")
    print(pqkd_results_df)
