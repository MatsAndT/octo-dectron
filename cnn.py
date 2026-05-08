from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from data.data import load_dronerf_dataframe
from mlp_infra import build_metrics_payload


TARGET_COLUMN_BY_HEAD = {
	"family": "target_family",
	"mode": "target_mode",
}

NUM_CLASSES_BY_HEAD = {
	"family": 4,
	"mode": 10,
}


@dataclass(slots=True)
class PreparedMultiTaskData:
	X_train: np.ndarray
	X_test: np.ndarray
	y_family_train: np.ndarray
	y_family_test: np.ndarray
	y_mode_train: np.ndarray
	y_mode_test: np.ndarray
	feature_columns: list[str]
	signal_mean: list[float] | None
	signal_std: list[float] | None


def mps_is_available() -> bool:
	return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def resolve_device(device_arg: str) -> torch.device:
	if device_arg == "auto":
		if torch.cuda.is_available():
			return torch.device("cuda")
		if mps_is_available():
			return torch.device("mps")
		return torch.device("cpu")

	if device_arg == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA was requested but is not available on this machine")
		return torch.device("cuda")

	if device_arg == "mps":
		if not mps_is_available():
			raise RuntimeError("MPS was requested but is not available on this machine")
		return torch.device("mps")

	return torch.device("cpu")


def set_global_seed(seed: int, deterministic: bool) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	if deterministic:
		torch.use_deterministic_algorithms(True, warn_only=True)
		if torch.backends.cudnn.is_available():
			torch.backends.cudnn.benchmark = False


def parse_conv_channels(raw_value: str) -> list[int]:
	channels: list[int] = []
	for part in raw_value.split(","):
		part = part.strip()
		if not part:
			continue
		value = int(part)
		if value <= 0:
			raise ValueError("Convolution channel sizes must be positive integers")
		channels.append(value)

	if not channels:
		raise ValueError("At least one convolution channel size must be provided")

	return channels


def resolve_window_settings(
	*,
	window_features_arg: bool | None,
	window_size: int,
	window_stride_arg: int | None,
) -> tuple[bool, int]:
	window_features = window_features_arg
	if window_features is None:
		# Multitask includes mode classification, so use windowing by default.
		window_features = True

	if window_stride_arg is not None:
		window_stride = window_stride_arg
	elif window_features:
		window_stride = max(1, window_size // 2)
	else:
		window_stride = window_size

	return bool(window_features), int(window_stride)


def _to_signal_array(values: Any) -> np.ndarray:
	array = np.asarray(values, dtype=np.float32).ravel()
	if array.size == 0:
		return np.array([], dtype=np.float32)
	if not np.all(np.isfinite(array)):
		array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
	return array


def _fit_signal_length(signal: np.ndarray, sequence_length: int) -> np.ndarray:
	if signal.size >= sequence_length:
		return signal[:sequence_length].astype(np.float32, copy=False)

	fitted = np.zeros(sequence_length, dtype=np.float32)
	fitted[: signal.size] = signal
	return fitted


def _build_signal_tensor(dataframe: pd.DataFrame, sequence_length: int) -> np.ndarray:
	rows = dataframe.shape[0]
	tensor = np.zeros((rows, 2, sequence_length), dtype=np.float32)

	for row_index, row in enumerate(dataframe.itertuples(index=False)):
		h_signal = _fit_signal_length(_to_signal_array(getattr(row, "h_raw_signal")), sequence_length)
		l_signal = _fit_signal_length(_to_signal_array(getattr(row, "l_raw_signal")), sequence_length)
		tensor[row_index, 0, :] = h_signal
		tensor[row_index, 1, :] = l_signal

	return tensor


def _resolve_stratify(
	y: np.ndarray,
	*,
	stratify: bool,
	test_size: float,
) -> tuple[np.ndarray | None, float]:
	effective_test_size = float(test_size)
	stratify_values = y if stratify and np.unique(y).size > 1 else None

	if stratify_values is None:
		return None, effective_test_size

	class_count = int(np.unique(y).size)
	sample_count = int(y.size)

	# Stratified split needs enough samples to place every class in both partitions.
	if sample_count < (2 * class_count):
		warnings.warn(
			"Stratified split disabled: not enough samples to place every class in "
			"both train and test sets. Falling back to non-stratified split.",
			RuntimeWarning,
		)
		return None, effective_test_size

	min_test_fraction = class_count / sample_count
	max_test_fraction = 1.0 - min_test_fraction

	if effective_test_size < min_test_fraction:
		warnings.warn(
			"Adjusted test_size from "
			f"{effective_test_size:.4f} to {min_test_fraction:.4f} "
			"to satisfy stratified split class coverage.",
			RuntimeWarning,
		)
		effective_test_size = min_test_fraction

	if effective_test_size > max_test_fraction:
		warnings.warn(
			"Adjusted test_size from "
			f"{effective_test_size:.4f} to {max_test_fraction:.4f} "
			"to keep stratified split feasible.",
			RuntimeWarning,
		)
		effective_test_size = max_test_fraction

	return stratify_values, effective_test_size


def prepare_multitask_data(
	dataframe: pd.DataFrame,
	*,
	sequence_length: int,
	test_size: float,
	random_state: int,
	stratify: bool = True,
	normalize_signals: bool = True,
) -> PreparedMultiTaskData:
	required_columns = [
		"h_raw_signal",
		"l_raw_signal",
		TARGET_COLUMN_BY_HEAD["family"],
		TARGET_COLUMN_BY_HEAD["mode"],
	]
	missing_columns = [column for column in required_columns if column not in dataframe.columns]
	if missing_columns:
		raise ValueError(
			"DataFrame does not include required multitask CNN columns. Missing: "
			f"{missing_columns}. Use load_dronerf_dataframe(..., include_raw_signal=True)."
		)

	if sequence_length <= 0:
		raise ValueError("sequence_length must be a positive integer")

	if not 0.0 < test_size < 1.0:
		raise ValueError("test_size must be between 0 and 1")

	X = _build_signal_tensor(dataframe=dataframe, sequence_length=sequence_length)
	y_family = dataframe[TARGET_COLUMN_BY_HEAD["family"]].to_numpy(dtype=np.int64)
	y_mode = dataframe[TARGET_COLUMN_BY_HEAD["mode"]].to_numpy(dtype=np.int64)

	try:
		model_selection = importlib.import_module("sklearn.model_selection")
		train_test_split = getattr(model_selection, "train_test_split")
	except ModuleNotFoundError as exc:
		raise RuntimeError(
			"scikit-learn is required for splitting. Install with `pip install scikit-learn`."
		) from exc

	stratify_values, effective_test_size = _resolve_stratify(
		y_mode,
		stratify=stratify,
		test_size=test_size,
	)

	(
		X_train,
		X_test,
		y_family_train,
		y_family_test,
		y_mode_train,
		y_mode_test,
	) = train_test_split(
		X,
		y_family,
		y_mode,
		test_size=effective_test_size,
		random_state=random_state,
		stratify=stratify_values,
	)

	signal_mean: list[float] | None = None
	signal_std: list[float] | None = None

	if normalize_signals:
		channel_mean = np.mean(X_train, axis=(0, 2), keepdims=True, dtype=np.float64)
		channel_std = np.std(X_train, axis=(0, 2), keepdims=True, dtype=np.float64)
		channel_std = np.where(channel_std < 1e-6, 1.0, channel_std)

		X_train = ((X_train - channel_mean) / channel_std).astype(np.float32)
		X_test = ((X_test - channel_mean) / channel_std).astype(np.float32)

		signal_mean = np.asarray(channel_mean, dtype=np.float64).reshape(-1).tolist()
		signal_std = np.asarray(channel_std, dtype=np.float64).reshape(-1).tolist()
	else:
		X_train = X_train.astype(np.float32)
		X_test = X_test.astype(np.float32)

	return PreparedMultiTaskData(
		X_train=X_train,
		X_test=X_test,
		y_family_train=y_family_train,
		y_family_test=y_family_test,
		y_mode_train=y_mode_train,
		y_mode_test=y_mode_test,
		feature_columns=["h_raw_signal", "l_raw_signal"],
		signal_mean=signal_mean,
		signal_std=signal_std,
	)


def make_multitask_loader(
	X: np.ndarray,
	y_family: np.ndarray,
	y_mode: np.ndarray,
	batch_size: int,
	shuffle: bool,
) -> DataLoader:
	features = torch.tensor(X, dtype=torch.float32)
	targets_family = torch.tensor(y_family, dtype=torch.long)
	targets_mode = torch.tensor(y_mode, dtype=torch.long)
	dataset = TensorDataset(features, targets_family, targets_mode)
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_class_weights(
	y: np.ndarray,
	num_classes: int,
	device: torch.device,
) -> torch.Tensor:
	counts = np.bincount(y, minlength=num_classes).astype(np.float64)
	weights = np.zeros(num_classes, dtype=np.float32)
	non_zero = counts > 0

	# Inverse-frequency weighting to reduce majority-class bias.
	weights[non_zero] = counts.sum() / (num_classes * counts[non_zero])

	return torch.tensor(weights, dtype=torch.float32, device=device)


class DroneCNNMultiTask(nn.Module):
	def __init__(
		self,
		*,
		input_channels: int,
		conv_channels: list[int],
		kernel_size: int,
		dropout: float,
		family_classes: int,
		mode_classes: int,
	) -> None:
		super().__init__()

		if input_channels <= 0:
			raise ValueError("input_channels must be positive")
		if not conv_channels:
			raise ValueError("conv_channels must include at least one layer")
		if kernel_size <= 0:
			raise ValueError("kernel_size must be positive")

		padding = kernel_size // 2
		layers: list[nn.Module] = []
		in_channels = input_channels

		for out_channels in conv_channels:
			layers.append(
				nn.Conv1d(
					in_channels=in_channels,
					out_channels=out_channels,
					kernel_size=kernel_size,
					padding=padding,
				)
			)
			layers.append(nn.BatchNorm1d(out_channels))
			layers.append(nn.ReLU())
			layers.append(nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True))
			if dropout > 0:
				layers.append(nn.Dropout(dropout * 0.5))
			in_channels = out_channels

		self.encoder = nn.Sequential(*layers)
		self.pool = nn.AdaptiveAvgPool1d(1)
		self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
		self.family_head = nn.Linear(in_channels, family_classes)
		self.mode_head = nn.Linear(in_channels, mode_classes)

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		features = self.encoder(x)
		features = self.pool(features).squeeze(-1)
		features = self.dropout(features)
		return self.family_head(features), self.mode_head(features)


def evaluate_model(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	family_criterion: nn.Module,
	mode_criterion: nn.Module,
	*,
	family_loss_weight: float,
	mode_loss_weight: float,
) -> dict[str, Any]:
	model.eval()
	total_loss = 0.0
	total_family_loss = 0.0
	total_mode_loss = 0.0

	family_true_values: list[int] = []
	family_pred_values: list[int] = []
	mode_true_values: list[int] = []
	mode_pred_values: list[int] = []

	with torch.no_grad():
		for features, family_targets, mode_targets in loader:
			features = features.to(device)
			family_targets = family_targets.to(device)
			mode_targets = mode_targets.to(device)

			family_logits, mode_logits = model(features)
			family_loss = family_criterion(family_logits, family_targets)
			mode_loss = mode_criterion(mode_logits, mode_targets)
			combined_loss = (family_loss_weight * family_loss) + (mode_loss_weight * mode_loss)

			batch_size = family_targets.size(0)
			total_loss += combined_loss.item() * batch_size
			total_family_loss += family_loss.item() * batch_size
			total_mode_loss += mode_loss.item() * batch_size

			family_predictions = torch.argmax(family_logits, dim=1)
			mode_predictions = torch.argmax(mode_logits, dim=1)

			family_true_values.extend(family_targets.cpu().numpy().tolist())
			family_pred_values.extend(family_predictions.cpu().numpy().tolist())
			mode_true_values.extend(mode_targets.cpu().numpy().tolist())
			mode_pred_values.extend(mode_predictions.cpu().numpy().tolist())

	family_y_true = np.asarray(family_true_values, dtype=np.int64)
	family_y_pred = np.asarray(family_pred_values, dtype=np.int64)
	mode_y_true = np.asarray(mode_true_values, dtype=np.int64)
	mode_y_pred = np.asarray(mode_pred_values, dtype=np.int64)

	if family_y_true.size == 0:
		return {
			"loss": 0.0,
			"family_loss": 0.0,
			"mode_loss": 0.0,
			"family_accuracy": 0.0,
			"mode_accuracy": 0.0,
			"family_macro_f1": 0.0,
			"mode_macro_f1": 0.0,
			"combined_macro_f1": 0.0,
			"family_y_true": family_y_true,
			"family_y_pred": family_y_pred,
			"mode_y_true": mode_y_true,
			"mode_y_pred": mode_y_pred,
		}

	family_accuracy = float(accuracy_score(family_y_true, family_y_pred))
	mode_accuracy = float(accuracy_score(mode_y_true, mode_y_pred))
	family_macro_f1 = float(f1_score(family_y_true, family_y_pred, average="macro", zero_division=0))
	mode_macro_f1 = float(f1_score(mode_y_true, mode_y_pred, average="macro", zero_division=0))

	return {
		"loss": total_loss / family_y_true.size,
		"family_loss": total_family_loss / family_y_true.size,
		"mode_loss": total_mode_loss / family_y_true.size,
		"family_accuracy": family_accuracy,
		"mode_accuracy": mode_accuracy,
		"family_macro_f1": family_macro_f1,
		"mode_macro_f1": mode_macro_f1,
		"combined_macro_f1": (family_macro_f1 + mode_macro_f1) / 2.0,
		"family_y_true": family_y_true,
		"family_y_pred": family_y_pred,
		"mode_y_true": mode_y_true,
		"mode_y_pred": mode_y_pred,
	}


def train_model(
	model: nn.Module,
	train_loader: DataLoader,
	val_loader: DataLoader,
	family_criterion: nn.Module,
	mode_criterion: nn.Module,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	epochs: int,
	early_stopping_patience: int,
	*,
	family_loss_weight: float,
	mode_loss_weight: float,
) -> tuple[list[dict[str, float]], int, dict[str, Any]]:
	history: list[dict[str, float]] = []
	best_epoch = 0
	best_score = -1.0
	best_state: dict[str, torch.Tensor] | None = None
	patience_counter = 0

	for epoch in range(1, epochs + 1):
		model.train()
		total_loss = 0.0
		total_family_loss = 0.0
		total_mode_loss = 0.0
		sample_count = 0

		for features, family_targets, mode_targets in train_loader:
			features = features.to(device)
			family_targets = family_targets.to(device)
			mode_targets = mode_targets.to(device)

			optimizer.zero_grad()
			family_logits, mode_logits = model(features)
			family_loss = family_criterion(family_logits, family_targets)
			mode_loss = mode_criterion(mode_logits, mode_targets)
			combined_loss = (family_loss_weight * family_loss) + (mode_loss_weight * mode_loss)
			combined_loss.backward()
			optimizer.step()

			batch_size = family_targets.size(0)
			total_loss += combined_loss.item() * batch_size
			total_family_loss += family_loss.item() * batch_size
			total_mode_loss += mode_loss.item() * batch_size
			sample_count += batch_size

		train_loss = total_loss / max(sample_count, 1)
		train_family_loss = total_family_loss / max(sample_count, 1)
		train_mode_loss = total_mode_loss / max(sample_count, 1)

		val_metrics = evaluate_model(
			model=model,
			loader=val_loader,
			device=device,
			family_criterion=family_criterion,
			mode_criterion=mode_criterion,
			family_loss_weight=family_loss_weight,
			mode_loss_weight=mode_loss_weight,
		)

		history.append(
			{
				"epoch": float(epoch),
				"train_loss": float(train_loss),
				"train_family_loss": float(train_family_loss),
				"train_mode_loss": float(train_mode_loss),
				"val_loss": float(val_metrics["loss"]),
				"val_family_loss": float(val_metrics["family_loss"]),
				"val_mode_loss": float(val_metrics["mode_loss"]),
				"val_family_accuracy": float(val_metrics["family_accuracy"]),
				"val_mode_accuracy": float(val_metrics["mode_accuracy"]),
				"val_family_macro_f1": float(val_metrics["family_macro_f1"]),
				"val_mode_macro_f1": float(val_metrics["mode_macro_f1"]),
				"val_combined_macro_f1": float(val_metrics["combined_macro_f1"]),
			}
		)

		print(
			f"Epoch {epoch:03d}/{epochs} "
			f"train_loss={train_loss:.4f} "
			f"val_loss={val_metrics['loss']:.4f} "
			f"family_acc={val_metrics['family_accuracy']:.4f} "
			f"family_f1={val_metrics['family_macro_f1']:.4f} "
			f"mode_acc={val_metrics['mode_accuracy']:.4f} "
			f"mode_f1={val_metrics['mode_macro_f1']:.4f}"
		)

		if val_metrics["combined_macro_f1"] > best_score:
			best_score = float(val_metrics["combined_macro_f1"])
			best_epoch = epoch
			best_state = {
				name: param.detach().cpu().clone()
				for name, param in model.state_dict().items()
			}
			patience_counter = 0
		else:
			patience_counter += 1
			if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
				print(
					"Early stopping triggered after "
					f"{early_stopping_patience} epochs without improvement"
				)
				break

	if best_state is not None:
		model.load_state_dict(best_state)

	best_metrics = evaluate_model(
		model=model,
		loader=val_loader,
		device=device,
		family_criterion=family_criterion,
		mode_criterion=mode_criterion,
		family_loss_weight=family_loss_weight,
		mode_loss_weight=mode_loss_weight,
	)
	return history, best_epoch, best_metrics


def _head_history(history: list[dict[str, float]], head: str) -> list[dict[str, float]]:
	return [
		{
			"epoch": float(item["epoch"]),
			"train_loss": float(item[f"train_{head}_loss"]),
			"val_loss": float(item[f"val_{head}_loss"]),
			"val_accuracy": float(item[f"val_{head}_accuracy"]),
			"val_macro_f1": float(item[f"val_{head}_macro_f1"]),
		}
		for item in history
	]


def _save_head_outputs(
	*,
	output_dir: Path,
	file_prefix: str,
	head: str,
	num_classes: int,
	y_true: np.ndarray,
	y_pred: np.ndarray,
	seed: int,
	device_type: str,
	X_train: np.ndarray,
	X_test: np.ndarray,
	y_train: np.ndarray,
	y_test: np.ndarray,
	best_epoch: int,
	head_loss: float,
	head_accuracy: float,
	head_macro_f1: float,
	history: list[dict[str, float]],
	classification_report_payload: dict[str, Any],
	model_path: Path,
	window_features: bool,
	window_size: int,
	window_stride: int,
	sequence_length: int,
) -> tuple[Path, Path, Path]:
	confusion_path = output_dir / f"{file_prefix}_confusion_{head}.csv"
	predictions_path = output_dir / f"{file_prefix}_predictions_{head}.csv"
	metrics_path = output_dir / f"{file_prefix}_metrics_{head}.json"

	conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
	np.savetxt(confusion_path, conf_matrix, delimiter=",", fmt="%d")

	predictions_array = np.column_stack((y_true, y_pred))
	np.savetxt(
		predictions_path,
		predictions_array,
		delimiter=",",
		fmt="%d",
		header="y_true,y_pred",
		comments="",
	)

	metrics_payload = build_metrics_payload(
		target=head,
		target_column=TARGET_COLUMN_BY_HEAD[head],
		num_classes=num_classes,
		seed=seed,
		device_type=device_type,
		X_train=X_train,
		X_test=X_test,
		best_epoch=best_epoch,
		best_metrics={
			"loss": head_loss,
			"accuracy": head_accuracy,
			"macro_f1": head_macro_f1,
		},
		history=history,
		classification_report_payload=classification_report_payload,
		y_train=y_train,
		y_test=y_test,
	)
	metrics_payload["multitask"] = True
	metrics_payload["sample_by_window"] = bool(window_features)
	metrics_payload["window_size"] = int(window_size) if window_features else None
	metrics_payload["window_stride"] = int(window_stride) if window_features else None
	metrics_payload["sequence_length"] = int(sequence_length)
	metrics_payload["model_path"] = str(model_path)

	metrics_path.write_text(json.dumps(metrics_payload, indent=2))
	return metrics_path, confusion_path, predictions_path


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train a multitask CNN classifier on DroneRF")

	parser.add_argument("--data-root", type=Path, default=Path("data/DroneRF"))
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--epochs", type=int, default=50)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--conv-channels", type=str, default="32,64,128")
	parser.add_argument("--kernel-size", type=int, default=7)
	parser.add_argument("--dropout", type=float, default=0.30)
	parser.add_argument("--test-size", type=float, default=0.20)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
	parser.add_argument("--model-dir", type=Path, default=Path("models"))
	parser.add_argument("--output-dir", type=Path, default=Path("results"))
	parser.add_argument("--max-values-per-archive", type=int, default=200_000)
	parser.add_argument("--fft-bins", type=int, default=2048)
	parser.add_argument("--window-size", type=int, default=4096)
	parser.add_argument("--window-stride", type=int, default=None)
	parser.add_argument("--sequence-length", type=int, default=None)
	parser.add_argument("--loader-cache-dir", type=Path, default=None)
	parser.add_argument("--loader-max-workers", type=int, default=0)
	parser.add_argument("--early-stopping-patience", type=int, default=10)
	parser.add_argument("--family-loss-weight", type=float, default=1.0)
	parser.add_argument("--mode-loss-weight", type=float, default=1.0)
	parser.add_argument(
		"--auto-test",
		action=argparse.BooleanOptionalAction,
		default=False,
	)
	parser.add_argument(
		"--window-features",
		action=argparse.BooleanOptionalAction,
		default=None,
	)

	parser.add_argument(
		"--extract-archives",
		action=argparse.BooleanOptionalAction,
		default=True,
	)
	parser.add_argument(
		"--force-reextract",
		action=argparse.BooleanOptionalAction,
		default=False,
	)
	parser.add_argument(
		"--normalize-signals",
		action=argparse.BooleanOptionalAction,
		default=True,
	)
	parser.add_argument(
		"--use-class-weights",
		action=argparse.BooleanOptionalAction,
		default=True,
	)
	parser.add_argument(
		"--deterministic",
		action=argparse.BooleanOptionalAction,
		default=False,
	)
	parser.add_argument(
		"--show-loading-progress",
		action=argparse.BooleanOptionalAction,
		default=True,
	)
	parser.add_argument(
		"--use-loader-cache",
		action=argparse.BooleanOptionalAction,
		default=True,
	)
	parser.add_argument(
		"--refresh-loader-cache",
		action=argparse.BooleanOptionalAction,
		default=False,
	)

	return parser


def main() -> None:
	args = build_arg_parser().parse_args()

	conv_channels = parse_conv_channels(args.conv_channels)
	device = resolve_device(args.device)

	if args.kernel_size <= 0:
		raise ValueError("kernel_size must be a positive integer")
	if args.family_loss_weight <= 0.0 or args.mode_loss_weight <= 0.0:
		raise ValueError("family_loss_weight and mode_loss_weight must be > 0")

	set_global_seed(args.seed, args.deterministic)
	loader_max_workers = args.loader_max_workers if args.loader_max_workers > 0 else None
	window_features, window_stride = resolve_window_settings(
		window_features_arg=args.window_features,
		window_size=args.window_size,
		window_stride_arg=args.window_stride,
	)

	sequence_length = args.sequence_length if args.sequence_length is not None else args.window_size
	if sequence_length <= 0:
		raise ValueError("sequence_length must be a positive integer")

	args.model_dir.mkdir(parents=True, exist_ok=True)
	args.output_dir.mkdir(parents=True, exist_ok=True)

	print("Loading DroneRF data...")
	dataframe = load_dronerf_dataframe(
		data_root=args.data_root,
		extract_archives=args.extract_archives,
		force_reextract=args.force_reextract,
		max_values_per_archive=args.max_values_per_archive,
		fft_bins=args.fft_bins,
		sample_by_window=window_features,
		window_size=args.window_size,
		window_stride=window_stride,
		show_progress=args.show_loading_progress,
		use_cache=args.use_loader_cache,
		refresh_cache=args.refresh_loader_cache,
		cache_dir=args.loader_cache_dir,
		max_workers=loader_max_workers,
		include_raw_signal=True,
	)

	prepared = prepare_multitask_data(
		dataframe=dataframe,
		sequence_length=sequence_length,
		test_size=args.test_size,
		random_state=args.seed,
		stratify=True,
		normalize_signals=args.normalize_signals,
	)

	train_loader = make_multitask_loader(
		prepared.X_train,
		prepared.y_family_train,
		prepared.y_mode_train,
		batch_size=args.batch_size,
		shuffle=True,
	)
	test_loader = make_multitask_loader(
		prepared.X_test,
		prepared.y_family_test,
		prepared.y_mode_test,
		batch_size=args.batch_size,
		shuffle=False,
	)

	model = DroneCNNMultiTask(
		input_channels=2,
		conv_channels=conv_channels,
		kernel_size=args.kernel_size,
		dropout=args.dropout,
		family_classes=NUM_CLASSES_BY_HEAD["family"],
		mode_classes=NUM_CLASSES_BY_HEAD["mode"],
	).to(device)

	family_weights: torch.Tensor | None = None
	mode_weights: torch.Tensor | None = None
	if args.use_class_weights:
		family_weights = compute_class_weights(
			prepared.y_family_train,
			NUM_CLASSES_BY_HEAD["family"],
			device,
		)
		mode_weights = compute_class_weights(
			prepared.y_mode_train,
			NUM_CLASSES_BY_HEAD["mode"],
			device,
		)

	family_criterion = nn.CrossEntropyLoss(weight=family_weights)
	mode_criterion = nn.CrossEntropyLoss(weight=mode_weights)
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=args.learning_rate,
		weight_decay=args.weight_decay,
	)

	print(
		f"Training on {device.type} with {prepared.X_train.shape[0]} train samples and "
		f"{prepared.X_test.shape[0]} test samples"
	)
	print(
		f"Input shape=(batch, channels=2, sequence_length={sequence_length}), "
		f"conv_channels={conv_channels}, kernel_size={args.kernel_size}"
	)
	if window_features:
		print(
			"Using window-based raw signals "
			f"(window_size={args.window_size}, window_stride={window_stride})"
		)

	history, best_epoch, best_metrics = train_model(
		model=model,
		train_loader=train_loader,
		val_loader=test_loader,
		family_criterion=family_criterion,
		mode_criterion=mode_criterion,
		optimizer=optimizer,
		device=device,
		epochs=args.epochs,
		early_stopping_patience=args.early_stopping_patience,
		family_loss_weight=args.family_loss_weight,
		mode_loss_weight=args.mode_loss_weight,
	)

	model_path = args.model_dir / "cnn_multitask.pt"
	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"model_name": "DroneCNNMultiTask",
			"num_classes_by_head": NUM_CLASSES_BY_HEAD,
			"target_columns": TARGET_COLUMN_BY_HEAD,
			"input_channels": 2,
			"sequence_length": int(sequence_length),
			"conv_channels": conv_channels,
			"kernel_size": int(args.kernel_size),
			"dropout": float(args.dropout),
			"feature_columns": prepared.feature_columns,
			"sample_by_window": bool(window_features),
			"window_size": int(args.window_size) if window_features else None,
			"window_stride": int(window_stride) if window_features else None,
			"normalize_signals": bool(args.normalize_signals),
			"signal_mean": prepared.signal_mean,
			"signal_std": prepared.signal_std,
			"family_loss_weight": float(args.family_loss_weight),
			"mode_loss_weight": float(args.mode_loss_weight),
		},
		model_path,
	)

	head_outputs: dict[str, tuple[Path, Path, Path]] = {}
	for head in ("family", "mode"):
		num_classes = NUM_CLASSES_BY_HEAD[head]
		y_true = cast(np.ndarray, best_metrics[f"{head}_y_true"])
		y_pred = cast(np.ndarray, best_metrics[f"{head}_y_pred"])
		report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
		report_payload = cast(dict[str, Any], report)

		head_outputs[head] = _save_head_outputs(
			output_dir=args.output_dir,
			file_prefix="cnn",
			head=head,
			num_classes=num_classes,
			y_true=y_true,
			y_pred=y_pred,
			seed=args.seed,
			device_type=device.type,
			X_train=prepared.X_train,
			X_test=prepared.X_test,
			y_train=prepared.y_family_train if head == "family" else prepared.y_mode_train,
			y_test=prepared.y_family_test if head == "family" else prepared.y_mode_test,
			best_epoch=best_epoch,
			head_loss=float(best_metrics[f"{head}_loss"]),
			head_accuracy=float(best_metrics[f"{head}_accuracy"]),
			head_macro_f1=float(best_metrics[f"{head}_macro_f1"]),
			history=_head_history(history, head),
			classification_report_payload=report_payload,
			model_path=model_path,
			window_features=window_features,
			window_size=args.window_size,
			window_stride=window_stride,
			sequence_length=sequence_length,
		)

	print(f"Saved model: {model_path}")
	for head, (metrics_path, confusion_path, predictions_path) in head_outputs.items():
		print(f"Saved {head} metrics: {metrics_path}")
		print(f"Saved {head} confusion matrix: {confusion_path}")
		print(f"Saved {head} predictions: {predictions_path}")

	print(
		"Validation accuracy: "
		f"family={best_metrics['family_accuracy']:.4f}, "
		f"mode={best_metrics['mode_accuracy']:.4f}, "
		f"combined_macro_f1={best_metrics['combined_macro_f1']:.4f}"
	)

	if args.auto_test:
		print("Running final auto-test pass...")
		test_metrics = evaluate_model(
			model=model,
			loader=test_loader,
			device=device,
			family_criterion=family_criterion,
			mode_criterion=mode_criterion,
			family_loss_weight=args.family_loss_weight,
			mode_loss_weight=args.mode_loss_weight,
		)

		for head in ("family", "mode"):
			num_classes = NUM_CLASSES_BY_HEAD[head]
			y_true = cast(np.ndarray, test_metrics[f"{head}_y_true"])
			y_pred = cast(np.ndarray, test_metrics[f"{head}_y_pred"])

			test_confusion_path = args.output_dir / f"cnn_test_confusion_{head}.csv"
			test_predictions_path = args.output_dir / f"cnn_test_predictions_{head}.csv"
			test_metrics_path = args.output_dir / f"cnn_test_metrics_{head}.json"

			test_conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
			np.savetxt(test_confusion_path, test_conf_matrix, delimiter=",", fmt="%d")

			test_predictions_array = np.column_stack((y_true, y_pred))
			np.savetxt(
				test_predictions_path,
				test_predictions_array,
				delimiter=",",
				fmt="%d",
				header="y_true,y_pred",
				comments="",
			)

			test_report = classification_report(
				y_true,
				y_pred,
				output_dict=True,
				zero_division=0,
			)
			test_payload = {
				"target": head,
				"loss": float(test_metrics[f"{head}_loss"]),
				"accuracy": float(test_metrics[f"{head}_accuracy"]),
				"macro_f1": float(test_metrics[f"{head}_macro_f1"]),
				"combined_macro_f1": float(test_metrics["combined_macro_f1"]),
				"classification_report": cast(dict[str, Any], test_report),
			}
			test_metrics_path.write_text(json.dumps(test_payload, indent=2))

			print(
				f"Auto-test {head}: "
				f"acc={test_payload['accuracy']:.4f} "
				f"macro_f1={test_payload['macro_f1']:.4f} "
				f"loss={test_payload['loss']:.4f}"
			)
			print(f"Saved auto-test {head} metrics: {test_metrics_path}")
			print(f"Saved auto-test {head} confusion matrix: {test_confusion_path}")
			print(f"Saved auto-test {head} predictions: {test_predictions_path}")


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("\nInterrupted by user (Ctrl+C). Exiting cleanly.", file=sys.stderr, flush=True)
		raise SystemExit(130)
