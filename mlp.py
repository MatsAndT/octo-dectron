from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from data.data import load_dronerf_dataframe
from mlp_infra import build_metrics_payload, prepare_training_data, serialize_scaler


TARGET_COLUMN_BY_NAME = {
	"binary": "target_binary",
	"family": "target_family",
	"mode": "target_mode",
}

NUM_CLASSES_BY_TARGET = {
	"binary": 2,
	"family": 4,
	"mode": 10,
}


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


def resolve_window_settings(
	*,
	target: str,
	window_features_arg: bool | None,
	window_size: int,
	window_stride_arg: int | None,
) -> tuple[bool, int]:
	window_features = window_features_arg
	if window_features is None:
		# Mode classification is data-starved with one row per archive pair,
		# so use windowed features by default for this target.
		window_features = target == "mode"

	if window_stride_arg is not None:
		window_stride = window_stride_arg
	elif window_features and target == "mode":
		window_stride = max(1, window_size // 2)
	else:
		window_stride = window_size

	return bool(window_features), int(window_stride)


class DroneMLP(nn.Module):
	def __init__(
		self,
		input_size: int,
		hidden_sizes: list[int],
		num_classes: int,
		dropout: float,
	) -> None:
		super().__init__()

		layers: list[nn.Module] = []
		in_features = input_size

		for hidden_size in hidden_sizes:
			layers.append(nn.Linear(in_features, hidden_size))
			layers.append(nn.ReLU())
			if dropout > 0:
				layers.append(nn.Dropout(dropout))
			in_features = hidden_size

		layers.append(nn.Linear(in_features, num_classes))
		self.network = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.network(x)


def parse_hidden_sizes(raw_value: str) -> list[int]:
	hidden_sizes: list[int] = []
	for part in raw_value.split(","):
		part = part.strip()
		if not part:
			continue
		value = int(part)
		if value <= 0:
			raise ValueError("Hidden layer sizes must be positive integers")
		hidden_sizes.append(value)

	if not hidden_sizes:
		raise ValueError("At least one hidden layer size must be provided")

	return hidden_sizes


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


def make_loader(
	X: np.ndarray,
	y: np.ndarray,
	batch_size: int,
	shuffle: bool,
) -> DataLoader:
	features = torch.tensor(X, dtype=torch.float32)
	targets = torch.tensor(y, dtype=torch.long)
	dataset = TensorDataset(features, targets)
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


def evaluate_model(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	criterion: nn.Module,
) -> dict[str, Any]:
	model.eval()
	total_loss = 0.0
	all_true: list[int] = []
	all_pred: list[int] = []

	with torch.no_grad():
		for features, targets in loader:
			features = features.to(device)
			targets = targets.to(device)

			logits = model(features)
			loss = criterion(logits, targets)
			total_loss += loss.item() * targets.size(0)

			predictions = torch.argmax(logits, dim=1)
			all_true.extend(targets.cpu().numpy().tolist())
			all_pred.extend(predictions.cpu().numpy().tolist())

	y_true = np.asarray(all_true, dtype=np.int64)
	y_pred = np.asarray(all_pred, dtype=np.int64)

	if y_true.size == 0:
		return {
			"loss": 0.0,
			"accuracy": 0.0,
			"macro_f1": 0.0,
			"y_true": y_true,
			"y_pred": y_pred,
		}

	return {
		"loss": total_loss / y_true.size,
		"accuracy": accuracy_score(y_true, y_pred),
		"macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
		"y_true": y_true,
		"y_pred": y_pred,
	}


def train_model(
	model: nn.Module,
	train_loader: DataLoader,
	val_loader: DataLoader,
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	epochs: int,
	early_stopping_patience: int,
) -> tuple[list[dict[str, float]], int, dict[str, Any]]:
	history: list[dict[str, float]] = []
	best_epoch = 0
	best_score = -1.0
	best_state: dict[str, torch.Tensor] | None = None
	patience_counter = 0

	for epoch in range(1, epochs + 1):
		model.train()
		running_loss = 0.0
		sample_count = 0

		for features, targets in train_loader:
			features = features.to(device)
			targets = targets.to(device)

			optimizer.zero_grad()
			logits = model(features)
			loss = criterion(logits, targets)
			loss.backward()
			optimizer.step()

			running_loss += loss.item() * targets.size(0)
			sample_count += targets.size(0)

		train_loss = running_loss / max(sample_count, 1)
		val_metrics = evaluate_model(model, val_loader, device, criterion)

		history.append(
			{
				"epoch": float(epoch),
				"train_loss": float(train_loss),
				"val_loss": float(val_metrics["loss"]),
				"val_accuracy": float(val_metrics["accuracy"]),
				"val_macro_f1": float(val_metrics["macro_f1"]),
			}
		)

		print(
			f"Epoch {epoch:03d}/{epochs} "
			f"train_loss={train_loss:.4f} "
			f"val_loss={val_metrics['loss']:.4f} "
			f"val_acc={val_metrics['accuracy']:.4f} "
			f"val_macro_f1={val_metrics['macro_f1']:.4f}"
		)

		if val_metrics["macro_f1"] > best_score:
			best_score = float(val_metrics["macro_f1"])
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

	best_metrics = evaluate_model(model, val_loader, device, criterion)
	return history, best_epoch, best_metrics


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train an MLP classifier on DroneRF")

	parser.add_argument("--data-root", type=Path, default=Path("data/DroneRF"))
	parser.add_argument("--target", choices=["binary", "family", "mode"], default="family")
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--epochs", type=int, default=50)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--hidden-sizes", type=str, default="256,128")
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
	parser.add_argument("--loader-cache-dir", type=Path, default=None)
	parser.add_argument("--loader-max-workers", type=int, default=0)
	parser.add_argument("--early-stopping-patience", type=int, default=10)
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
		"--scale-features",
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

	hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
	target_column = TARGET_COLUMN_BY_NAME[args.target]
	num_classes = NUM_CLASSES_BY_TARGET[args.target]

	device = resolve_device(args.device)

	set_global_seed(args.seed, args.deterministic)
	loader_max_workers = args.loader_max_workers if args.loader_max_workers > 0 else None
	window_features, window_stride = resolve_window_settings(
		target=args.target,
		window_features_arg=args.window_features,
		window_size=args.window_size,
		window_stride_arg=args.window_stride,
	)

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
	)

	prepared = prepare_training_data(
		dataframe=dataframe,
		target_column=target_column,
		test_size=args.test_size,
		random_state=args.seed,
		stratify=True,
		scale_features=args.scale_features,
	)

	X_train = prepared.X_train
	X_test = prepared.X_test
	y_train = prepared.y_train
	y_test = prepared.y_test
	scaler = prepared.scaler

	train_loader = make_loader(X_train, y_train, batch_size=args.batch_size, shuffle=True)
	test_loader = make_loader(X_test, y_test, batch_size=args.batch_size, shuffle=False)

	model = DroneMLP(
		input_size=X_train.shape[1],
		hidden_sizes=hidden_sizes,
		num_classes=num_classes,
		dropout=args.dropout,
	).to(device)

	class_weights: torch.Tensor | None = None
	if args.use_class_weights:
		class_weights = compute_class_weights(y_train, num_classes, device)

	criterion = nn.CrossEntropyLoss(weight=class_weights)
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=args.learning_rate,
		weight_decay=args.weight_decay,
	)

	print(
		f"Training on {device.type} with {X_train.shape[0]} train samples and "
		f"{X_test.shape[0]} test samples"
	)
	if window_features:
		print(
			"Using window-based features "
			f"(window_size={args.window_size}, window_stride={window_stride})"
		)

	history, best_epoch, best_metrics = train_model(
		model=model,
		train_loader=train_loader,
		val_loader=test_loader,
		criterion=criterion,
		optimizer=optimizer,
		device=device,
		epochs=args.epochs,
		early_stopping_patience=args.early_stopping_patience,
	)

	y_true = best_metrics["y_true"]
	y_pred = best_metrics["y_pred"]

	conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
	report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
	report_payload = cast(dict[str, Any], report)

	model_path = args.model_dir / f"mlp_{args.target}.pt"
	metrics_path = args.output_dir / f"metrics_{args.target}.json"
	confusion_path = args.output_dir / f"confusion_{args.target}.csv"
	predictions_path = args.output_dir / f"predictions_{args.target}.csv"

	scaler_mean, scaler_scale = serialize_scaler(scaler)

	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"target": args.target,
			"target_column": target_column,
			"num_classes": num_classes,
			"input_size": int(X_train.shape[1]),
			"hidden_sizes": hidden_sizes,
			"dropout": float(args.dropout),
			"feature_columns": prepared.feature_columns,
			"sample_by_window": bool(window_features),
			"window_size": int(args.window_size) if window_features else None,
			"window_stride": int(window_stride) if window_features else None,
			"scaler_mean": scaler_mean,
			"scaler_scale": scaler_scale,
		},
		model_path,
	)

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
		target=args.target,
		target_column=target_column,
		num_classes=num_classes,
		seed=args.seed,
		device_type=device.type,
		X_train=X_train,
		X_test=X_test,
		best_epoch=best_epoch,
		best_metrics=best_metrics,
		history=history,
		classification_report_payload=report_payload,
		y_train=y_train,
		y_test=y_test,
	)
	metrics_payload["sample_by_window"] = bool(window_features)
	metrics_payload["window_size"] = int(args.window_size) if window_features else None
	metrics_payload["window_stride"] = int(window_stride) if window_features else None

	metrics_path.write_text(json.dumps(metrics_payload, indent=2))

	print(f"Saved model: {model_path}")
	print(f"Saved metrics: {metrics_path}")
	print(f"Saved confusion matrix: {confusion_path}")
	print(f"Saved predictions: {predictions_path}")

	if args.auto_test:
		print("Running final auto-test pass...")
		test_metrics = evaluate_model(model, test_loader, device, criterion)
		test_y_true = test_metrics["y_true"]
		test_y_pred = test_metrics["y_pred"]

		test_confusion_path = args.output_dir / f"test_confusion_{args.target}.csv"
		test_predictions_path = args.output_dir / f"test_predictions_{args.target}.csv"
		test_metrics_path = args.output_dir / f"test_metrics_{args.target}.json"

		test_conf_matrix = confusion_matrix(
			test_y_true,
			test_y_pred,
			labels=list(range(num_classes)),
		)
		np.savetxt(test_confusion_path, test_conf_matrix, delimiter=",", fmt="%d")

		test_predictions_array = np.column_stack((test_y_true, test_y_pred))
		np.savetxt(
			test_predictions_path,
			test_predictions_array,
			delimiter=",",
			fmt="%d",
			header="y_true,y_pred",
			comments="",
		)

		test_report = classification_report(
			test_y_true,
			test_y_pred,
			output_dict=True,
			zero_division=0,
		)
		test_payload = {
			"target": args.target,
			"loss": float(test_metrics["loss"]),
			"accuracy": float(test_metrics["accuracy"]),
			"macro_f1": float(test_metrics["macro_f1"]),
			"classification_report": cast(dict[str, Any], test_report),
		}
		test_metrics_path.write_text(json.dumps(test_payload, indent=2))

		print(
			"Auto-test metrics: "
			f"acc={test_payload['accuracy']:.4f} "
			f"macro_f1={test_payload['macro_f1']:.4f} "
			f"loss={test_payload['loss']:.4f}"
		)
		print(f"Saved auto-test metrics: {test_metrics_path}")
		print(f"Saved auto-test confusion matrix: {test_confusion_path}")
		print(f"Saved auto-test predictions: {test_predictions_path}")


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("\nInterrupted by user (Ctrl+C). Exiting cleanly.", file=sys.stderr, flush=True)
		raise SystemExit(130)
