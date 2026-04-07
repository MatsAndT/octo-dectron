from __future__ import annotations

import numpy as np

from data.data import _iter_signal_windows, _window_feature_rows


def test_iter_signal_windows_covers_tail() -> None:
	signal = np.arange(10, dtype=np.float64)
	windows = _iter_signal_windows(signal=signal, window_size=4, window_stride=4)

	starts = [start for start, _ in windows]
	lengths = [window.size for _, window in windows]

	assert starts == [0, 4, 6]
	assert lengths == [4, 4, 4]


def test_window_feature_rows_expected_count_and_keys() -> None:
	signal = np.linspace(0.0, 1.0, 20, dtype=np.float64)
	rows = _window_feature_rows(signal=signal, fft_bins=8, window_size=5, window_stride=5)

	assert len(rows) == 4
	for index, row in enumerate(rows):
		assert row["window_index"] == index
		assert row["window_length"] == 5
		assert "mean" in row
		assert "fft_entropy" in row


def test_window_feature_rows_short_signal_single_window() -> None:
	signal = np.array([1.0, 2.0, 3.0], dtype=np.float64)
	rows = _window_feature_rows(signal=signal, fft_bins=8, window_size=16, window_stride=8)

	assert len(rows) == 1
	assert rows[0]["window_start"] == 0
	assert rows[0]["window_length"] == 3
