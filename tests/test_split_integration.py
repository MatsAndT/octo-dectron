from __future__ import annotations

import os
from pathlib import Path

import pytest

from data.data import dataframe_to_numpy, load_dronerf_dataframe


@pytest.mark.integration
def test_real_dronerf_split_optional() -> None:
	if os.environ.get("OCTO_RUN_INTEGRATION") != "1":
		pytest.skip("Set OCTO_RUN_INTEGRATION=1 to run real-data integration tests")

	data_root = Path("data/DroneRF")
	if not data_root.exists():
		pytest.skip("DroneRF dataset root not found")

	dataframe = load_dronerf_dataframe(
		data_root=data_root,
		extract_archives=True,
		max_values_per_archive=50_000,
		show_progress=False,
		use_cache=True,
		max_workers=1,
	)

	arrays = dataframe_to_numpy(
		dataframe=dataframe,
		target_column="target_family",
		test_size=0.2,
		random_state=42,
		stratify=True,
	)

	assert arrays["X_train"].shape[0] > 0
	assert arrays["X_test"].shape[0] > 0
	assert arrays["X_train"].shape[1] == arrays["X_test"].shape[1]