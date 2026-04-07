from __future__ import annotations

import numpy as np
import pandas as pd

from mlp_infra import build_metrics_payload, prepare_training_data, serialize_scaler


def _build_dataframe(rows_per_class: int = 10) -> pd.DataFrame:
	rng = np.random.default_rng(42)
	rows: list[dict[str, float | int]] = []

	for label in range(2):
		for _ in range(rows_per_class):
			rows.append(
				{
					"f1": float(rng.normal(loc=label)),
					"f2": float(rng.normal(loc=label)),
					"target_family": label,
				}
			)

	return pd.DataFrame(rows)


def test_prepare_training_data_with_scaling() -> None:
	df = _build_dataframe(rows_per_class=8)

	prepared = prepare_training_data(
		dataframe=df,
		target_column="target_family",
		test_size=0.25,
		random_state=42,
		feature_columns=["f1", "f2"],
		stratify=True,
		scale_features=True,
	)

	assert prepared.X_train.shape[0] == 12
	assert prepared.X_test.shape[0] == 4
	assert prepared.X_train.dtype == np.float32
	assert prepared.X_test.dtype == np.float32
	assert prepared.scaler is not None
	assert prepared.feature_columns == ["f1", "f2"]


def test_prepare_training_data_without_scaling() -> None:
	df = _build_dataframe(rows_per_class=6)

	prepared = prepare_training_data(
		dataframe=df,
		target_column="target_family",
		test_size=0.25,
		random_state=0,
		feature_columns=["f1", "f2"],
		stratify=True,
		scale_features=False,
	)

	assert prepared.scaler is None
	assert prepared.X_train.dtype == np.float32


def test_serialize_scaler_handles_none() -> None:
	mean, scale = serialize_scaler(None)
	assert mean is None
	assert scale is None


def test_build_metrics_payload_structure() -> None:
	X_train = np.zeros((6, 2), dtype=np.float32)
	X_test = np.zeros((2, 2), dtype=np.float32)
	y_train = np.array([0, 0, 1, 1, 1, 0], dtype=np.int64)
	y_test = np.array([0, 1], dtype=np.int64)

	payload = build_metrics_payload(
		target="family",
		target_column="target_family",
		num_classes=2,
		seed=123,
		device_type="cpu",
		X_train=X_train,
		X_test=X_test,
		best_epoch=2,
		best_metrics={"loss": 1.1, "accuracy": 0.5, "macro_f1": 0.5},
		history=[{"epoch": 1.0, "train_loss": 1.0, "val_loss": 1.1, "val_accuracy": 0.5, "val_macro_f1": 0.5}],
		classification_report_payload={"accuracy": 0.5},
		y_train=y_train,
		y_test=y_test,
	)

	assert payload["target"] == "family"
	assert payload["target_column"] == "target_family"
	assert payload["train_samples"] == 6
	assert payload["test_samples"] == 2
	assert payload["class_distribution_train"] == [3, 3]
	assert payload["class_distribution_test"] == [1, 1]