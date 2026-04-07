from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.data import dataframe_to_numpy


@dataclass(slots=True)
class PreparedTrainingData:
	X_train: np.ndarray
	X_test: np.ndarray
	y_train: np.ndarray
	y_test: np.ndarray
	feature_columns: list[str]
	scaler: StandardScaler | None


def prepare_training_data(
	dataframe: pd.DataFrame,
	*,
	target_column: str,
	test_size: float,
	random_state: int,
	feature_columns: Sequence[str] | None = None,
	stratify: bool = True,
	scale_features: bool = True,
) -> PreparedTrainingData:
	"""
	Convert DataFrame to train/test arrays with optional feature scaling.
	"""

	arrays = dataframe_to_numpy(
		dataframe=dataframe,
		target_column=target_column,
		feature_columns=feature_columns,
		test_size=test_size,
		random_state=random_state,
		stratify=stratify,
	)

	X_train = arrays["X_train"]
	X_test = arrays["X_test"]
	y_train = arrays["y_train"]
	y_test = arrays["y_test"]

	scaler: StandardScaler | None = None
	if scale_features:
		scaler = StandardScaler()
		X_train = scaler.fit_transform(X_train).astype(np.float32)
		X_test = scaler.transform(X_test).astype(np.float32)
	else:
		X_train = X_train.astype(np.float32)
		X_test = X_test.astype(np.float32)

	return PreparedTrainingData(
		X_train=X_train,
		X_test=X_test,
		y_train=y_train,
		y_test=y_test,
		feature_columns=list(arrays["feature_columns"]),
		scaler=scaler,
	)


def serialize_scaler(scaler: StandardScaler | None) -> tuple[list[float] | None, list[float] | None]:
	if scaler is None or scaler.mean_ is None or scaler.scale_ is None:
		return None, None

	return (
		np.asarray(scaler.mean_, dtype=np.float64).tolist(),
		np.asarray(scaler.scale_, dtype=np.float64).tolist(),
	)


def build_metrics_payload(
	*,
	target: str,
	target_column: str,
	num_classes: int,
	seed: int,
	device_type: str,
	X_train: np.ndarray,
	X_test: np.ndarray,
	best_epoch: int,
	best_metrics: dict[str, Any],
	history: list[dict[str, float]],
	classification_report_payload: dict[str, Any],
	y_train: np.ndarray,
	y_test: np.ndarray,
) -> dict[str, Any]:
	return {
		"target": target,
		"target_column": target_column,
		"num_classes": num_classes,
		"seed": seed,
		"device": device_type,
		"train_samples": int(X_train.shape[0]),
		"test_samples": int(X_test.shape[0]),
		"best_epoch": int(best_epoch),
		"best_loss": float(best_metrics["loss"]),
		"best_accuracy": float(best_metrics["accuracy"]),
		"best_macro_f1": float(best_metrics["macro_f1"]),
		"history": history,
		"classification_report": classification_report_payload,
		"class_distribution_train": np.bincount(y_train, minlength=num_classes).tolist(),
		"class_distribution_test": np.bincount(y_test, minlength=num_classes).tolist(),
	}
