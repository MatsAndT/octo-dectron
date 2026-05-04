from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cnn import (
	DroneCNNMultiTask,
	build_arg_parser,
	compute_class_weights,
	evaluate_model,
	make_multitask_loader,
	parse_conv_channels,
	prepare_multitask_data,
	resolve_window_settings,
	train_model,
)


def test_parse_conv_channels_valid() -> None:
	assert parse_conv_channels("32, 64,128") == [32, 64, 128]


def test_parse_conv_channels_invalid() -> None:
	with pytest.raises(ValueError):
		parse_conv_channels("  ")

	with pytest.raises(ValueError):
		parse_conv_channels("16,-4")


def test_auto_test_flag_parsing() -> None:
	parser = build_arg_parser()

	default_args = parser.parse_args([])
	assert default_args.auto_test is False

	enabled_args = parser.parse_args(["--auto-test"])
	assert enabled_args.auto_test is True

	disabled_args = parser.parse_args(["--no-auto-test"])
	assert disabled_args.auto_test is False


def test_window_feature_flags_parsing() -> None:
	parser = build_arg_parser()

	default_args = parser.parse_args([])
	assert default_args.window_features is None
	assert default_args.window_size == 4096
	assert default_args.window_stride is None

	enabled_args = parser.parse_args(["--window-features", "--window-size", "2048", "--window-stride", "1024"])
	assert enabled_args.window_features is True
	assert enabled_args.window_size == 2048
	assert enabled_args.window_stride == 1024

	disabled_args = parser.parse_args(["--no-window-features"])
	assert disabled_args.window_features is False


def test_resolve_window_settings_defaults() -> None:
	window_features, window_stride = resolve_window_settings(
		window_features_arg=None,
		window_size=4096,
		window_stride_arg=None,
	)
	assert window_features is True
	assert window_stride == 2048


def test_prepare_multitask_data_shapes() -> None:
	dataframe = pd.DataFrame(
		{
			"h_raw_signal": [
				np.array([1.0, 2.0, 3.0], dtype=np.float32),
				np.array([4.0, 5.0], dtype=np.float32),
				np.array([6.0, 7.0, 8.0, 9.0], dtype=np.float32),
				np.array([1.5], dtype=np.float32),
			],
			"l_raw_signal": [
				np.array([0.5, 0.2], dtype=np.float32),
				np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
				np.array([0.7], dtype=np.float32),
				np.array([1.0, 2.0, 3.0], dtype=np.float32),
			],
			"target_family": [0, 1, 2, 3],
			"target_mode": [0, 1, 5, 9],
		}
	)

	prepared = prepare_multitask_data(
		dataframe=dataframe,
		sequence_length=8,
		test_size=0.25,
		random_state=42,
		stratify=False,
		normalize_signals=False,
	)

	assert prepared.X_train.shape[1:] == (2, 8)
	assert prepared.X_test.shape[1:] == (2, 8)
	assert prepared.X_train.dtype == np.float32
	assert prepared.X_test.dtype == np.float32
	assert prepared.y_family_train.dtype == np.int64
	assert prepared.y_mode_test.dtype == np.int64
	assert np.isfinite(prepared.X_train).all()
	assert np.isfinite(prepared.X_test).all()


def test_prepare_multitask_data_requires_raw_signals() -> None:
	dataframe = pd.DataFrame(
		{
			"target_family": [0, 1],
			"target_mode": [0, 1],
		}
	)

	with pytest.raises(ValueError, match="include_raw_signal"):
		prepare_multitask_data(
			dataframe=dataframe,
			sequence_length=16,
			test_size=0.5,
			random_state=0,
			stratify=False,
			normalize_signals=False,
		)


def test_dronemultitask_forward_shape() -> None:
	model = DroneCNNMultiTask(
		input_channels=2,
		conv_channels=[8, 16],
		kernel_size=5,
		dropout=0.1,
		family_classes=4,
		mode_classes=10,
	)
	x = torch.randn(6, 2, 128)
	family_logits, mode_logits = model(x)

	assert family_logits.shape == (6, 4)
	assert mode_logits.shape == (6, 10)


def test_compute_class_weights_inverse_frequency() -> None:
	y = np.array([0, 0, 0, 1], dtype=np.int64)
	weights = compute_class_weights(y=y, num_classes=2, device=torch.device("cpu"))

	assert weights.shape == (2,)
	assert torch.isclose(weights[0], torch.tensor(2.0 / 3.0), atol=1e-6)
	assert torch.isclose(weights[1], torch.tensor(2.0), atol=1e-6)


def test_evaluate_model_with_empty_loader_returns_zero_metrics() -> None:
	model = DroneCNNMultiTask(
		input_channels=2,
		conv_channels=[8],
		kernel_size=3,
		dropout=0.0,
		family_classes=4,
		mode_classes=10,
	)
	family_criterion = nn.CrossEntropyLoss()
	mode_criterion = nn.CrossEntropyLoss()

	features = torch.empty((0, 2, 64), dtype=torch.float32)
	targets_family = torch.empty((0,), dtype=torch.long)
	targets_mode = torch.empty((0,), dtype=torch.long)
	loader: DataLoader = DataLoader(
		TensorDataset(features, targets_family, targets_mode),
		batch_size=4,
		shuffle=False,
	)

	metrics = evaluate_model(
		model=model,
		loader=loader,
		device=torch.device("cpu"),
		family_criterion=family_criterion,
		mode_criterion=mode_criterion,
		family_loss_weight=1.0,
		mode_loss_weight=1.0,
	)

	assert metrics["loss"] == 0.0
	assert metrics["family_accuracy"] == 0.0
	assert metrics["mode_accuracy"] == 0.0
	assert metrics["combined_macro_f1"] == 0.0
	assert metrics["family_y_true"].size == 0
	assert metrics["mode_y_pred"].size == 0


def test_train_model_returns_history_and_metrics() -> None:
	rng = np.random.default_rng(1)
	X = rng.normal(size=(30, 2, 64)).astype(np.float32)
	y_family = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int64)
	y_mode = np.array([0, 1, 2, 3, 4] * 6, dtype=np.int64)

	train_loader = make_multitask_loader(
		X[:20],
		y_family[:20],
		y_mode[:20],
		batch_size=8,
		shuffle=True,
	)
	val_loader = make_multitask_loader(
		X[20:],
		y_family[20:],
		y_mode[20:],
		batch_size=8,
		shuffle=False,
	)

	model = DroneCNNMultiTask(
		input_channels=2,
		conv_channels=[8, 16],
		kernel_size=5,
		dropout=0.1,
		family_classes=4,
		mode_classes=10,
	)
	family_criterion = nn.CrossEntropyLoss()
	mode_criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

	history, best_epoch, best_metrics = train_model(
		model=model,
		train_loader=train_loader,
		val_loader=val_loader,
		family_criterion=family_criterion,
		mode_criterion=mode_criterion,
		optimizer=optimizer,
		device=torch.device("cpu"),
		epochs=3,
		early_stopping_patience=2,
		family_loss_weight=1.0,
		mode_loss_weight=1.0,
	)

	assert len(history) >= 1
	assert 1 <= best_epoch <= len(history)
	assert "family_accuracy" in best_metrics
	assert "mode_macro_f1" in best_metrics
	assert best_metrics["family_y_true"].size > 0
