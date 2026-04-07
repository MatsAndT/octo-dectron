from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.data import dataframe_to_numpy


def _make_dataframe(class_counts: list[int]) -> pd.DataFrame:
	rng = np.random.default_rng(7)
	rows: list[dict[str, float | int]] = []

	for class_id, count in enumerate(class_counts):
		for _ in range(count):
			rows.append(
				{
					"f1": float(rng.normal(loc=class_id)),
					"f2": float(rng.normal(loc=class_id)),
					"target_family": class_id,
				}
			)

	return pd.DataFrame(rows)


def test_stratified_split_auto_adjusts_small_test_fraction() -> None:
	# 12 samples, 4 classes -> min test fraction for stratification is 4/12.
	df = _make_dataframe([3, 3, 3, 3])

	with pytest.warns(RuntimeWarning, match="Adjusted test_size"):
		arrays = dataframe_to_numpy(
			dataframe=df,
			target_column="target_family",
			feature_columns=["f1", "f2"],
			test_size=0.2,
			random_state=42,
			stratify=True,
		)

	assert arrays["X_test"].shape[0] == 4
	assert arrays["X_train"].shape[0] == 8
	assert set(np.unique(arrays["y_test"])) == {0, 1, 2, 3}


def test_stratified_split_falls_back_when_not_feasible() -> None:
	# 6 samples, 4 classes -> cannot place every class in both train and test.
	df = _make_dataframe([2, 2, 1, 1])

	with pytest.warns(RuntimeWarning, match="Stratified split disabled"):
		arrays = dataframe_to_numpy(
			dataframe=df,
			target_column="target_family",
			feature_columns=["f1", "f2"],
			test_size=0.2,
			random_state=123,
			stratify=True,
		)

	assert arrays["X_train"].shape[0] + arrays["X_test"].shape[0] == 6
	assert arrays["y_train"].dtype == np.int64


def test_dataframe_to_numpy_without_split_returns_full_arrays() -> None:
	df = _make_dataframe([4, 4])
	arrays = dataframe_to_numpy(
		dataframe=df,
		target_column="target_family",
		feature_columns=["f1", "f2"],
		test_size=None,
		random_state=0,
		stratify=True,
	)

	assert "X_train" not in arrays
	assert arrays["X"].shape == (8, 2)
	assert arrays["y"].shape == (8,)