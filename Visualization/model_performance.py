from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass(slots=True)
class MetricsBundle:
	label: str
	target: str
	train_loss: float | None
	train_accuracy: float | None
	train_macro_f1: float | None
	test_loss: float | None
	test_accuracy: float | None
	test_macro_f1: float | None


@dataclass(slots=True)
class ConfusionBundle:
	label: str
	target: str
	matrix: np.ndarray
	variant: str


def _load_json(path: Path) -> dict[str, Any] | None:
	if not path.exists():
		return None
	try:
		return json.loads(path.read_text())
	except Exception:
		return None


def _safe_float(value: Any) -> float | None:
	try:
		return float(value)
	except (TypeError, ValueError):
		return None


def _extract_metrics(payload: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
	return (
		_safe_float(payload.get("loss")),
		_safe_float(payload.get("accuracy")),
		_safe_float(payload.get("macro_f1")),
	)


def _load_metrics_bundle(
	*,
	label: str,
	target: str,
	train_path: Path,
	test_path: Path | None,
) -> MetricsBundle | None:
	train_payload = _load_json(train_path)
	if train_payload is None:
		return None

	if "best_loss" in train_payload:
		train_loss = _safe_float(train_payload.get("best_loss"))
		train_accuracy = _safe_float(train_payload.get("best_accuracy"))
		train_macro_f1 = _safe_float(train_payload.get("best_macro_f1"))
	else:
		train_loss, train_accuracy, train_macro_f1 = _extract_metrics(train_payload)

	test_loss = None
	test_accuracy = None
	test_macro_f1 = None
	if test_path is not None:
		test_payload = _load_json(test_path)
		if test_payload is not None:
			test_loss, test_accuracy, test_macro_f1 = _extract_metrics(test_payload)

	return MetricsBundle(
		label=label,
		target=target,
		train_loss=train_loss,
		train_accuracy=train_accuracy,
		train_macro_f1=train_macro_f1,
		test_loss=test_loss,
		test_accuracy=test_accuracy,
		test_macro_f1=test_macro_f1,
	)


def _load_confusion_matrix(path: Path) -> np.ndarray | None:
	if not path.exists():
		return None
	try:
		with path.open(newline="", encoding="utf-8") as handle:
			reader = csv.reader(handle)
			rows = [[float(value) for value in row] for row in reader if row]
		if not rows:
			return None
		return np.asarray(rows, dtype=np.float64)
	except Exception:
		return None


def _gather_metrics(results_root: Path, targets: list[str], include_test: bool) -> list[MetricsBundle]:
	bundles: list[MetricsBundle] = []
	for target in targets:
		cnn_bundle = _load_metrics_bundle(
			label="CNN",
			target=target,
			train_path=results_root / f"cnn_metrics_{target}.json",
			test_path=(results_root / f"cnn_test_metrics_{target}.json") if include_test else None,
		)
		if cnn_bundle is not None:
			bundles.append(cnn_bundle)

		mlp_bundle = _load_metrics_bundle(
			label="MLP",
			target=target,
			train_path=results_root / f"metrics_{target}.json",
			test_path=(results_root / f"test_metrics_{target}.json") if include_test else None,
		)
		if mlp_bundle is not None:
			bundles.append(mlp_bundle)

	return bundles


def _gather_confusions(results_root: Path, targets: list[str], include_test: bool) -> list[ConfusionBundle]:
	bundles: list[ConfusionBundle] = []
	for target in targets:
		cnn_train = _load_confusion_matrix(results_root / f"cnn_confusion_{target}.csv")
		if cnn_train is not None:
			bundles.append(
				ConfusionBundle(label="CNN", target=target, matrix=cnn_train, variant="train")
			)

		cnn_test = (
			_load_confusion_matrix(results_root / f"cnn_test_confusion_{target}.csv")
			if include_test
			else None
		)
		if cnn_test is not None:
			bundles.append(
				ConfusionBundle(label="CNN", target=target, matrix=cnn_test, variant="test")
			)

		mlp_train = _load_confusion_matrix(results_root / f"confusion_{target}.csv")
		if mlp_train is not None:
			bundles.append(
				ConfusionBundle(label="MLP", target=target, matrix=mlp_train, variant="train")
			)

		mlp_test = (
			_load_confusion_matrix(results_root / f"test_confusion_{target}.csv")
			if include_test
			else None
		)
		if mlp_test is not None:
			bundles.append(
				ConfusionBundle(label="MLP", target=target, matrix=mlp_test, variant="test")
			)

	return bundles


def _plot_metric_bars(
	bundles: list[MetricsBundle],
	*,
	metric_name: str,
	metric_label: str,
	use_test: bool,
) -> None:
	labels: list[str] = []
	values: list[float] = []
	for bundle in bundles:
		value = (
			getattr(bundle, f"test_{metric_name}") if use_test else getattr(bundle, f"train_{metric_name}")
		)
		if value is None:
			continue
		labels.append(f"{bundle.label}-{bundle.target}")
		values.append(float(value))

	if not values:
		return

	fig, ax = plt.subplots(figsize=(10, 4))
	x = np.arange(len(values))
	ax.bar(x, values, color="#3b82f6" if use_test else "#10b981")
	ax.set_xticks(x)
	ax.set_xticklabels(labels, rotation=30, ha="right")
	ax.set_title(f"{metric_label} ({'test' if use_test else 'train'})")
	ax.set_ylabel(metric_label)
	ax.grid(axis="y", linestyle="--", alpha=0.5)
	fig.tight_layout()


def _plot_train_vs_test(bundles: list[MetricsBundle], metric_name: str, metric_label: str) -> None:
	labels: list[str] = []
	train_values: list[float] = []
	test_values: list[float] = []
	for bundle in bundles:
		train_value = getattr(bundle, f"train_{metric_name}")
		test_value = getattr(bundle, f"test_{metric_name}")
		if train_value is None and test_value is None:
			continue
		labels.append(f"{bundle.label}-{bundle.target}")
		train_values.append(float(train_value) if train_value is not None else 0.0)
		test_values.append(float(test_value) if test_value is not None else 0.0)

	if not labels:
		return

	x = np.arange(len(labels))
	width = 0.35
	fig, ax = plt.subplots(figsize=(10, 4))
	ax.bar(x - width / 2, train_values, width, label="train", color="#10b981")
	ax.bar(x + width / 2, test_values, width, label="test", color="#f97316")
	ax.set_xticks(x)
	ax.set_xticklabels(labels, rotation=30, ha="right")
	ax.set_title(f"Train vs Test {metric_label}")
	ax.set_ylabel(metric_label)
	ax.legend()
	ax.grid(axis="y", linestyle="--", alpha=0.5)
	fig.tight_layout()


def _plot_confusion_matrices(confusions: list[ConfusionBundle]) -> None:
	for bundle in confusions:
		fig, ax = plt.subplots(figsize=(4.8, 4.2))
		image = ax.imshow(bundle.matrix, cmap="viridis")
		ax.set_title(f"{bundle.label} {bundle.target} ({bundle.variant})")
		ax.set_xlabel("Predicted")
		ax.set_ylabel("True")
		fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
		fig.tight_layout()


def _report_missing(results_root: Path, targets: list[str], include_test: bool) -> None:
	paths: list[Path] = []
	for target in targets:
		paths.extend(
			[
				results_root / f"cnn_metrics_{target}.json",
				results_root / f"metrics_{target}.json",
				results_root / f"cnn_confusion_{target}.csv",
				results_root / f"confusion_{target}.csv",
			]
		)
		if include_test:
			paths.extend(
				[
					results_root / f"cnn_test_metrics_{target}.json",
					results_root / f"test_metrics_{target}.json",
					results_root / f"cnn_test_confusion_{target}.csv",
					results_root / f"test_confusion_{target}.csv",
				]
			)

	missing = [path for path in paths if not path.exists()]
	found = [path for path in paths if path.exists()]

	print(f"Found {len(found)} result files.")
	if missing:
		print("Missing files:")
		for path in missing:
			print(f"- {path}")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Visualize CNN and MLP performance metrics.")
	parser.add_argument(
		"--results-root",
		type=Path,
		default=Path("results"),
		help="Path to results folder.",
	)
	parser.add_argument(
		"--targets",
		type=str,
		default="family,mode",
		help="Comma-separated targets to plot (family,mode,binary).",
	)
	parser.add_argument(
		"--no-test",
		action="store_true",
		help="Skip test metrics and confusion matrices.",
	)
	parser.add_argument(
		"--no-accuracy",
		action="store_true",
		help="Skip accuracy bars.",
	)
	parser.add_argument(
		"--no-macro-f1",
		action="store_true",
		help="Skip macro F1 bars.",
	)
	parser.add_argument(
		"--no-loss",
		action="store_true",
		help="Skip loss bars.",
	)
	parser.add_argument(
		"--no-train-test",
		action="store_true",
		help="Skip train vs test comparison.",
	)
	parser.add_argument(
		"--no-confusion",
		action="store_true",
		help="Skip confusion matrix plots.",
	)
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()
	results_root = args.results_root
	targets = [target.strip() for target in args.targets.split(",") if target.strip()]
	include_test = not args.no_test

	_report_missing(results_root, targets, include_test)

	bundles = _gather_metrics(results_root, targets, include_test)
	confusions = _gather_confusions(results_root, targets, include_test)

	if not args.no_accuracy:
		_plot_metric_bars(bundles, metric_name="accuracy", metric_label="Accuracy", use_test=False)
		if include_test:
			_plot_metric_bars(bundles, metric_name="accuracy", metric_label="Accuracy", use_test=True)

	if not args.no_macro_f1:
		_plot_metric_bars(bundles, metric_name="macro_f1", metric_label="Macro F1", use_test=False)
		if include_test:
			_plot_metric_bars(bundles, metric_name="macro_f1", metric_label="Macro F1", use_test=True)

	if not args.no_loss:
		_plot_metric_bars(bundles, metric_name="loss", metric_label="Loss", use_test=False)
		if include_test:
			_plot_metric_bars(bundles, metric_name="loss", metric_label="Loss", use_test=True)

	if not args.no_train_test and include_test:
		_plot_train_vs_test(bundles, metric_name="accuracy", metric_label="Accuracy")
		_plot_train_vs_test(bundles, metric_name="macro_f1", metric_label="Macro F1")
		_plot_train_vs_test(bundles, metric_name="loss", metric_label="Loss")

	if not args.no_confusion:
		_plot_confusion_matrices(confusions)

	if not plt.get_fignums():
		print("No plots were generated. Check that result files exist.")
		return

	plt.show()


if __name__ == "__main__":
	main()
