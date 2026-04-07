from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mlp import (
	DroneMLP,
	build_arg_parser,
	compute_class_weights,
	evaluate_model,
	make_loader,
	parse_hidden_sizes,
	resolve_window_settings,
	train_model,
)


def test_parse_hidden_sizes_valid() -> None:
	assert parse_hidden_sizes("256, 128") == [256, 128]


def test_parse_hidden_sizes_invalid() -> None:
	with pytest.raises(ValueError):
		parse_hidden_sizes(" ")

	with pytest.raises(ValueError):
		parse_hidden_sizes("32,-8")


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
	mode_window_features, mode_window_stride = resolve_window_settings(
		target="mode",
		window_features_arg=None,
		window_size=4096,
		window_stride_arg=None,
	)
	assert mode_window_features is True
	assert mode_window_stride == 2048

	family_window_features, family_window_stride = resolve_window_settings(
		target="family",
		window_features_arg=None,
		window_size=4096,
		window_stride_arg=None,
	)
	assert family_window_features is False
	assert family_window_stride == 4096


def test_resolve_window_settings_explicit_overrides() -> None:
	window_features, window_stride = resolve_window_settings(
		target="mode",
		window_features_arg=True,
		window_size=4096,
		window_stride_arg=1024,
	)
	assert window_features is True
	assert window_stride == 1024


def test_dronemlp_forward_shape() -> None:
	model = DroneMLP(input_size=4, hidden_sizes=[8, 4], num_classes=3, dropout=0.0)
	x = torch.randn(5, 4)
	logits = model(x)
	assert logits.shape == (5, 3)


def test_compute_class_weights_inverse_frequency() -> None:
	y = np.array([0, 0, 0, 1], dtype=np.int64)
	weights = compute_class_weights(y=y, num_classes=2, device=torch.device("cpu"))

	assert weights.shape == (2,)
	assert torch.isclose(weights[0], torch.tensor(2.0 / 3.0), atol=1e-6)
	assert torch.isclose(weights[1], torch.tensor(2.0), atol=1e-6)


def test_evaluate_model_with_empty_loader_returns_zero_metrics() -> None:
	model = DroneMLP(input_size=2, hidden_sizes=[4], num_classes=2, dropout=0.0)
	criterion = nn.CrossEntropyLoss()

	features = torch.empty((0, 2), dtype=torch.float32)
	targets = torch.empty((0,), dtype=torch.long)
	loader: DataLoader = DataLoader(TensorDataset(features, targets), batch_size=4, shuffle=False)

	metrics = evaluate_model(
		model=model,
		loader=loader,
		device=torch.device("cpu"),
		criterion=criterion,
	)

	assert metrics["loss"] == 0.0
	assert metrics["accuracy"] == 0.0
	assert metrics["macro_f1"] == 0.0
	assert metrics["y_true"].size == 0
	assert metrics["y_pred"].size == 0


def test_train_model_returns_history_and_metrics() -> None:
	rng = np.random.default_rng(0)
	X = rng.normal(size=(24, 4)).astype(np.float32)
	y = np.array([0] * 12 + [1] * 12, dtype=np.int64)

	train_loader = make_loader(X[:16], y[:16], batch_size=8, shuffle=True)
	val_loader = make_loader(X[16:], y[16:], batch_size=8, shuffle=False)

	model = DroneMLP(input_size=4, hidden_sizes=[16], num_classes=2, dropout=0.1)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

	history, best_epoch, best_metrics = train_model(
		model=model,
		train_loader=train_loader,
		val_loader=val_loader,
		criterion=criterion,
		optimizer=optimizer,
		device=torch.device("cpu"),
		epochs=3,
		early_stopping_patience=2,
	)

	assert len(history) >= 1
	assert 1 <= best_epoch <= len(history)
	assert "accuracy" in best_metrics
	assert "macro_f1" in best_metrics
	assert best_metrics["y_true"].size > 0