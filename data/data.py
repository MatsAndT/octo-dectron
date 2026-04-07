from __future__ import annotations

import re
import shutil
import subprocess
import importlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


_ARCHIVE_NAME_RE = re.compile(
	r"(?i)(?:RF|FR)[ _-]*Data[_ -]*(?P<code>\d{5})_(?P<band>[HL])(?P<index>\d*)$"
)

_MODE_ID_BY_CODE: dict[str, int] = {
	"00000": 0,
	"10000": 1,
	"10001": 2,
	"10010": 3,
	"10011": 4,
	"10100": 5,
	"10101": 6,
	"10110": 7,
	"10111": 8,
	"11000": 9,
}

_MODE_NAME_BY_CODE: dict[str, str] = {
	"00000": "background",
	"10000": "bebop_off",
	"10001": "bebop_on_connected",
	"10010": "bebop_hover",
	"10011": "bebop_flying_video",
	"10100": "ar_off",
	"10101": "ar_on_connected",
	"10110": "ar_hover",
	"10111": "ar_flying_video",
	"11000": "phantom",
}

_FAMILY_NAME_BY_CODE: dict[str, str] = {
	"00000": "background",
	"10000": "bebop",
	"10001": "bebop",
	"10010": "bebop",
	"10011": "bebop",
	"10100": "ar",
	"10101": "ar",
	"10110": "ar",
	"10111": "ar",
	"11000": "phantom",
}

_FAMILY_ID_BY_NAME: dict[str, int] = {
	"background": 0,
	"bebop": 1,
	"ar": 2,
	"phantom": 3,
}

_BASE_FEATURE_COLUMNS: tuple[str, ...] = (
	"signal_length",
	"mean",
	"std",
	"min",
	"max",
	"median",
	"q25",
	"q75",
	"iqr",
	"rms",
	"abs_mean",
	"energy",
	"fft_peak_bin",
	"fft_peak_power",
	"fft_mean_power",
	"fft_entropy",
)

_NUMERIC_FILE_EXTENSIONS = {
	".csv",
	".txt",
	".dat",
	".tsv",
	".log",
	".npy",
	".npz",
	".mat",
}


@dataclass(frozen=True)
class DroneRFArchive:
	code: str
	band: str
	archive_index: int | None
	path: Path


def _default_data_root() -> Path:
	return Path(__file__).resolve().parent / "DroneRF"


def _labels_from_code(code: str) -> dict[str, Any]:
	if code not in _MODE_ID_BY_CODE:
		raise ValueError(f"Unsupported DroneRF code in filename: {code}")

	family_name = _FAMILY_NAME_BY_CODE[code]
	return {
		"target_binary": 0 if family_name == "background" else 1,
		"target_family": _FAMILY_ID_BY_NAME[family_name],
		"target_mode": _MODE_ID_BY_CODE[code],
		"family_name": family_name,
		"mode_name": _MODE_NAME_BY_CODE[code],
	}


def _sanitize_for_path(value: str) -> str:
	return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _parse_archive_name(path: Path) -> DroneRFArchive | None:
	match = _ARCHIVE_NAME_RE.search(path.stem)
	if not match:
		return None

	index_raw = match.group("index")
	archive_index = int(index_raw) if index_raw else None
	return DroneRFArchive(
		code=match.group("code"),
		band=match.group("band").upper(),
		archive_index=archive_index,
		path=path,
	)


def _discover_archives(data_root: Path) -> list[DroneRFArchive]:
	archives: list[DroneRFArchive] = []

	for path in sorted(data_root.rglob("*.rar")):
		parsed = _parse_archive_name(path)
		if parsed is None:
			continue
		archives.append(parsed)

	return archives


def _extract_with_rarfile(archive_path: Path, destination: Path) -> str | None:
	try:
		rarfile = importlib.import_module("rarfile")
	except ModuleNotFoundError:
		return "Python package 'rarfile' is not installed."

	try:
		with rarfile.RarFile(archive_path) as archive:
			archive.extractall(path=destination)
		return None
	except Exception as exc:  # pragma: no cover - backend specific errors
		return f"rarfile failed: {exc}"


def _extract_with_cli(archive_path: Path, destination: Path) -> str | None:
	cli_commands: list[list[str]] = []

	if shutil.which("unar"):
		cli_commands.append(["unar", "-q", "-o", str(destination), str(archive_path)])
	if shutil.which("unrar"):
		cli_commands.append(["unrar", "x", "-o+", str(archive_path), str(destination)])
	if shutil.which("bsdtar"):
		cli_commands.append(["bsdtar", "-xf", str(archive_path), "-C", str(destination)])

	if not cli_commands:
		return "No archive CLI backend found (unar/unrar/bsdtar)."

	cli_errors: list[str] = []
	for command in cli_commands:
		result = subprocess.run(command, capture_output=True, text=True, check=False)
		if result.returncode == 0:
			return None
		stderr = result.stderr.strip() or result.stdout.strip() or "unknown error"
		cli_errors.append(f"{' '.join(command[:2])}: {stderr}")

	return "; ".join(cli_errors)


def _extract_archive(
	archive: DroneRFArchive,
	extract_root: Path,
	force_reextract: bool,
) -> Path:
	destination = (
		extract_root
		/ f"{archive.code}_{archive.band}{archive.archive_index or 0}_{_sanitize_for_path(archive.path.stem)}"
	)

	if force_reextract and destination.exists():
		shutil.rmtree(destination)

	if destination.exists() and any(destination.rglob("*")):
		return destination

	destination.mkdir(parents=True, exist_ok=True)

	extraction_errors: list[str] = []

	rarfile_error = _extract_with_rarfile(archive.path, destination)
	if rarfile_error is None:
		return destination
	extraction_errors.append(rarfile_error)

	cli_error = _extract_with_cli(archive.path, destination)
	if cli_error is None:
		return destination
	extraction_errors.append(cli_error)

	raise RuntimeError(
		"Failed to extract archive "
		f"{archive.path}. Tried Python and CLI backends. "
		"Install one of these options and retry: "
		"`pip install rarfile` plus `brew install unar`, or `brew install unrar`. "
		f"Backend errors: {extraction_errors}"
	)


def _iter_numeric_files(folder: Path) -> list[Path]:
	files: list[Path] = []
	for file_path in sorted(folder.rglob("*")):
		if not file_path.is_file():
			continue
		if file_path.name.startswith("."):
			continue
		if file_path.suffix.lower() not in _NUMERIC_FILE_EXTENSIONS:
			continue
		files.append(file_path)
	return files


def _read_numeric_array(path: Path) -> np.ndarray:
	suffix = path.suffix.lower()

	if suffix == ".npy":
		array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).ravel()
		return array

	if suffix == ".npz":
		with np.load(path, allow_pickle=False) as archive:
			arrays = [
				np.asarray(value, dtype=np.float64).ravel()
				for value in archive.values()
				if np.asarray(value).size > 0
			]
		if not arrays:
			return np.array([], dtype=np.float64)
		return np.concatenate(arrays)

	if suffix == ".mat":
		try:
			scipy_io = importlib.import_module("scipy.io")
			loadmat = getattr(scipy_io, "loadmat")
		except ModuleNotFoundError as exc:
			raise RuntimeError(
				"Found .mat data file but scipy is missing. Install with `pip install scipy`."
			) from exc

		mat_data = loadmat(path)
		arrays: list[np.ndarray] = []
		for key, value in mat_data.items():
			if key.startswith("__"):
				continue
			value_array = np.asarray(value)
			if not np.issubdtype(value_array.dtype, np.number):
				continue
			arrays.append(np.asarray(value_array, dtype=np.float64).ravel())

		if not arrays:
			return np.array([], dtype=np.float64)
		return np.concatenate(arrays)

	try:
		dataframe = pd.read_csv(path, header=None, engine="python", sep=None)
		numeric_series = pd.to_numeric(
			pd.Series(dataframe.to_numpy().ravel()),
			errors="coerce",
		)
		numeric_values = np.asarray(numeric_series, dtype=np.float64)
		numeric_values = numeric_values[~np.isnan(numeric_values)]
		if numeric_values.size > 0:
			return numeric_values
	except Exception:
		pass

	text = path.read_text(errors="ignore")
	matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
	if not matches:
		return np.array([], dtype=np.float64)
	return np.asarray([float(value) for value in matches], dtype=np.float64)


def _read_archive_signal(
	extracted_folder: Path,
	max_values_per_archive: int | None,
) -> np.ndarray:
	numeric_files = _iter_numeric_files(extracted_folder)
	if not numeric_files:
		raise ValueError(
			f"No supported numeric files found in extracted folder: {extracted_folder}"
		)

	chunks: list[np.ndarray] = []
	consumed = 0

	for file_path in numeric_files:
		values = _read_numeric_array(file_path)
		if values.size == 0:
			continue

		if max_values_per_archive is not None:
			remaining = max_values_per_archive - consumed
			if remaining <= 0:
				break
			values = values[:remaining]

		chunks.append(values)
		consumed += values.size

	if not chunks:
		raise ValueError(
			f"Numeric files were found but none contained values in {extracted_folder}"
		)

	if len(chunks) == 1:
		return chunks[0]
	return np.concatenate(chunks)


def _signal_features(signal: np.ndarray, fft_bins: int) -> dict[str, float]:
	if signal.size == 0:
		raise ValueError("Cannot compute features from an empty signal")

	signal = np.asarray(signal, dtype=np.float64)

	q25, q75 = np.percentile(signal, [25, 75])
	rms = float(np.sqrt(np.mean(np.square(signal))))
	energy = float(np.mean(np.square(signal)))

	fft_values = np.fft.rfft(signal, n=fft_bins)
	power = np.square(np.abs(fft_values))
	power_sum = float(np.sum(power))

	if power_sum > 0.0:
		power_norm = power / power_sum
		fft_entropy = float(
			-np.sum(power_norm * np.log2(np.clip(power_norm, 1e-12, None)))
		)
	else:
		fft_entropy = 0.0

	peak_bin = int(np.argmax(power)) if power.size > 0 else 0

	return {
		"signal_length": float(signal.size),
		"mean": float(np.mean(signal)),
		"std": float(np.std(signal)),
		"min": float(np.min(signal)),
		"max": float(np.max(signal)),
		"median": float(np.median(signal)),
		"q25": float(q25),
		"q75": float(q75),
		"iqr": float(q75 - q25),
		"rms": rms,
		"abs_mean": float(np.mean(np.abs(signal))),
		"energy": energy,
		"fft_peak_bin": float(peak_bin),
		"fft_peak_power": float(power[peak_bin]) if power.size > 0 else 0.0,
		"fft_mean_power": float(np.mean(power)) if power.size > 0 else 0.0,
		"fft_entropy": fft_entropy,
	}


def _repeat_pairing(
	high_rows: Sequence[dict[str, Any]],
	low_rows: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
	if not high_rows or not low_rows:
		return []

	pair_count = max(len(high_rows), len(low_rows))
	return [
		(
			high_rows[index % len(high_rows)],
			low_rows[index % len(low_rows)],
		)
		for index in range(pair_count)
	]


def _pair_high_low_rows(
	high_rows: Sequence[dict[Any, Any]],
	low_rows: Sequence[dict[Any, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
	high_rows_normalized: list[dict[str, Any]] = [
		{str(key): value for key, value in row.items()} for row in high_rows
	]
	low_rows_normalized: list[dict[str, Any]] = [
		{str(key): value for key, value in row.items()} for row in low_rows
	]

	high_by_index: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
	low_by_index: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
	high_unindexed: list[dict[str, Any]] = []
	low_unindexed: list[dict[str, Any]] = []

	for row in high_rows_normalized:
		index = row["archive_index"]
		if index is None:
			high_unindexed.append(row)
			continue
		high_by_index[int(index)].append(row)

	for row in low_rows_normalized:
		index = row["archive_index"]
		if index is None:
			low_unindexed.append(row)
			continue
		low_by_index[int(index)].append(row)

	pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []

	overlapping_indexes = sorted(set(high_by_index).intersection(low_by_index))
	for index in overlapping_indexes:
		pairs.extend(_repeat_pairing(high_by_index[index], low_by_index[index]))

	remaining_high: list[dict[str, Any]] = []
	remaining_low: list[dict[str, Any]] = []

	for index, rows in high_by_index.items():
		if index not in overlapping_indexes:
			remaining_high.extend(rows)
	for index, rows in low_by_index.items():
		if index not in overlapping_indexes:
			remaining_low.extend(rows)

	remaining_high.extend(high_unindexed)
	remaining_low.extend(low_unindexed)

	pairs.extend(_repeat_pairing(remaining_high, remaining_low))
	return pairs


def _build_pair_dataframe(
	archive_dataframe: pd.DataFrame,
	include_raw_signal: bool,
) -> pd.DataFrame:
	rows: list[dict[str, Any]] = []

	grouped = archive_dataframe.groupby("code", sort=True)
	for code, group in grouped:
		high_frame = group[group["band"] == "H"].copy()
		low_frame = group[group["band"] == "L"].copy()

		high_rows = sorted(
			high_frame.to_dict("records"),
			key=lambda row: (
				int(row["archive_index"]) if row["archive_index"] is not None else 10**9,
				str(row["archive_name"]),
			),
		)
		low_rows = sorted(
			low_frame.to_dict("records"),
			key=lambda row: (
				int(row["archive_index"]) if row["archive_index"] is not None else 10**9,
				str(row["archive_name"]),
			),
		)

		if not high_rows or not low_rows:
			raise ValueError(
				"Cannot build sample pairs for code "
				f"{code}: missing {'H' if not high_rows else 'L'} band archives."
			)

		labels = _labels_from_code(str(code))
		pairs = _pair_high_low_rows(high_rows, low_rows)

		for pair_index, (high_row, low_row) in enumerate(pairs, start=1):
			row: dict[str, Any] = {
				"sample_id": f"{code}_{pair_index:03d}",
				"code": code,
				"family_name": labels["family_name"],
				"mode_name": labels["mode_name"],
				"target_binary": labels["target_binary"],
				"target_family": labels["target_family"],
				"target_mode": labels["target_mode"],
				"h_archive_name": high_row["archive_name"],
				"l_archive_name": low_row["archive_name"],
				"h_archive_index": high_row["archive_index"],
				"l_archive_index": low_row["archive_index"],
				"h_archive_path": high_row["archive_path"],
				"l_archive_path": low_row["archive_path"],
				"h_source_dir": high_row["source_dir"],
				"l_source_dir": low_row["source_dir"],
			}

			for feature_name in _BASE_FEATURE_COLUMNS:
				row[f"h_{feature_name}"] = high_row[feature_name]
				row[f"l_{feature_name}"] = low_row[feature_name]

			row["delta_rms"] = row["h_rms"] - row["l_rms"]
			row["ratio_rms"] = row["h_rms"] / (row["l_rms"] + 1e-12)

			if include_raw_signal:
				row["h_raw_signal"] = high_row["raw_signal"]
				row["l_raw_signal"] = low_row["raw_signal"]

			rows.append(row)

	dataframe = pd.DataFrame(rows)
	if dataframe.empty:
		raise ValueError("No DroneRF samples were produced after pairing H/L archives")

	return dataframe


def load_dronerf_dataframe(
	data_root: str | Path | None = None,
	extract_to: str | Path | None = None,
	*,
	extract_archives: bool = True,
	force_reextract: bool = False,
	max_values_per_archive: int | None = 200_000,
	fft_bins: int = 2048,
	include_raw_signal: bool = False,
) -> pd.DataFrame:
	"""
	Load DroneRF data into a training-ready pandas DataFrame.

	The loader extracts each .rar archive (if needed), reads numeric signal files,
	computes compact statistical + FFT features per archive, then pairs H/L bands
	into one row per sample.
	"""

	data_root_path = Path(data_root) if data_root is not None else _default_data_root()
	data_root_path = data_root_path.resolve()

	if not data_root_path.exists():
		raise FileNotFoundError(f"DroneRF data root does not exist: {data_root_path}")

	extract_root = (
		Path(extract_to).resolve()
		if extract_to is not None
		else (data_root_path / "_extracted").resolve()
	)
	extract_root.mkdir(parents=True, exist_ok=True)

	archives = _discover_archives(data_root_path)
	if not archives:
		raise FileNotFoundError(
			f"No DroneRF .rar files found under: {data_root_path}. "
			"Expected files named like 'RF Data_10000_H.rar'."
		)

	archive_rows: list[dict[str, Any]] = []
	for archive in archives:
		extraction_folder = (
			_extract_archive(
				archive=archive,
				extract_root=extract_root,
				force_reextract=force_reextract,
			)
			if extract_archives
			else (
				extract_root
				/ f"{archive.code}_{archive.band}{archive.archive_index or 0}_{_sanitize_for_path(archive.path.stem)}"
			)
		)

		if not extraction_folder.exists():
			raise FileNotFoundError(
				"Extraction folder not found. Either set extract_archives=True or "
				f"point extract_to to an existing folder. Missing: {extraction_folder}"
			)

		signal = _read_archive_signal(
			extraction_folder,
			max_values_per_archive=max_values_per_archive,
		)
		feature_map = _signal_features(signal=signal, fft_bins=fft_bins)

		archive_row: dict[str, Any] = {
			"code": archive.code,
			"band": archive.band,
			"archive_index": archive.archive_index,
			"archive_name": archive.path.name,
			"archive_path": str(archive.path),
			"source_dir": str(extraction_folder),
			**feature_map,
		}

		if include_raw_signal:
			archive_row["raw_signal"] = signal

		archive_rows.append(archive_row)

	archive_dataframe = pd.DataFrame(archive_rows)
	if archive_dataframe.empty:
		raise ValueError("No archive rows were generated from DroneRF input files")

	return _build_pair_dataframe(
		archive_dataframe=archive_dataframe,
		include_raw_signal=include_raw_signal,
	)


def default_feature_columns(dataframe: pd.DataFrame) -> list[str]:
	feature_columns = [
		*[f"h_{column}" for column in _BASE_FEATURE_COLUMNS],
		*[f"l_{column}" for column in _BASE_FEATURE_COLUMNS],
		"delta_rms",
		"ratio_rms",
	]

	missing = [column for column in feature_columns if column not in dataframe.columns]
	if missing:
		raise ValueError(
			"DataFrame does not include expected feature columns. Missing: "
			f"{missing}. Make sure it was created with load_dronerf_dataframe()."
		)

	return feature_columns


def dataframe_to_numpy(
	dataframe: pd.DataFrame,
	target_column: str = "target_family",
	feature_columns: Sequence[str] | None = None,
	*,
	test_size: float | None = 0.2,
	random_state: int = 42,
	stratify: bool = True,
) -> dict[str, Any]:
	"""
	Convert DroneRF DataFrame to NumPy arrays, optionally with train/test split.
	"""

	if target_column not in dataframe.columns:
		raise ValueError(f"Unknown target column: {target_column}")

	selected_features = list(feature_columns) if feature_columns else default_feature_columns(dataframe)
	missing = [column for column in selected_features if column not in dataframe.columns]
	if missing:
		raise ValueError(f"Requested feature columns do not exist in DataFrame: {missing}")

	X = dataframe[selected_features].to_numpy(dtype=np.float32)
	y = dataframe[target_column].to_numpy(dtype=np.int64)

	result: dict[str, Any] = {
		"X": X,
		"y": y,
		"feature_columns": selected_features,
		"target_column": target_column,
	}

	if test_size is None:
		return result

	if not 0.0 < test_size < 1.0:
		raise ValueError("test_size must be between 0 and 1")

	try:
		model_selection = importlib.import_module("sklearn.model_selection")
		train_test_split = getattr(model_selection, "train_test_split")
	except ModuleNotFoundError as exc:
		raise RuntimeError(
			"scikit-learn is required for splitting. Install with `pip install scikit-learn`."
		) from exc

	stratify_values = y if stratify and np.unique(y).size > 1 else None
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		random_state=random_state,
		stratify=stratify_values,
	)

	result.update(
		{
			"X_train": X_train,
			"X_test": X_test,
			"y_train": y_train,
			"y_test": y_test,
		}
	)
	return result


def dataframe_to_torch_dataset(
	dataframe: pd.DataFrame,
	target_column: str = "target_family",
	feature_columns: Sequence[str] | None = None,
):
	"""
	Convert DroneRF DataFrame into a torch TensorDataset.
	"""

	try:
		torch = importlib.import_module("torch")
	except ModuleNotFoundError as exc:
		raise RuntimeError("PyTorch is not installed. Use `pip install torch`.") from exc

	arrays = dataframe_to_numpy(
		dataframe=dataframe,
		target_column=target_column,
		feature_columns=feature_columns,
		test_size=None,
	)

	features_tensor = torch.tensor(arrays["X"], dtype=torch.float32)
	targets_tensor = torch.tensor(arrays["y"], dtype=torch.long)
	return torch.utils.data.TensorDataset(features_tensor, targets_tensor)


def dataframe_to_torch_dataloader(
	dataframe: pd.DataFrame,
	target_column: str = "target_family",
	feature_columns: Sequence[str] | None = None,
	*,
	batch_size: int = 32,
	shuffle: bool = True,
	num_workers: int = 0,
):
	"""
	Convert DroneRF DataFrame into a torch DataLoader.
	"""

	try:
		torch = importlib.import_module("torch")
	except ModuleNotFoundError as exc:
		raise RuntimeError("PyTorch is not installed. Use `pip install torch`.") from exc

	dataset = dataframe_to_torch_dataset(
		dataframe=dataframe,
		target_column=target_column,
		feature_columns=feature_columns,
	)
	return torch.utils.data.DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
	)


__all__ = [
	"load_dronerf_dataframe",
	"default_feature_columns",
	"dataframe_to_numpy",
	"dataframe_to_torch_dataset",
	"dataframe_to_torch_dataloader",
]
