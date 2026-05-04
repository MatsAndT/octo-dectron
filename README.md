# octo-dectron
Computer program to estimate what kind of drones are in the area, based on RX/TX communication.

## DEV
1. Install UV and run uv sync
```bash
uv sync
```
2. Install DroneRF from [https://data.mendeley.com/datasets/f4c2b4n755/1](https://data.mendeley.com/datasets/f4c2b4n755/1) and unzip the file. Put it in data/ and call it `DroneRF`

## DroneRF Loader

Use the loader in [data/data.py](data/data.py) to extract `.rar` files and build a training-ready DataFrame. You need to first download the DroneRF datasett from 

```python
from data.data import load_dronerf_dataframe, dataframe_to_numpy

# 1) Load and extract DroneRF archives into a feature DataFrame
df = load_dronerf_dataframe()

# 2) Convert to NumPy arrays for sklearn/Keras training
arrays = dataframe_to_numpy(df, target_column="target_family", test_size=0.2)
X_train, y_train = arrays["X_train"], arrays["y_train"]
X_test, y_test = arrays["X_test"], arrays["y_test"]
```

### Label Targets

- `target_binary`: background vs drone
- `target_family`: background, bebop, ar, phantom
- `target_mode`: fine-grained 10-class label from DroneRF code

### Optional PyTorch

```python
from data.data import load_dronerf_dataframe, dataframe_to_torch_dataloader

df = load_dronerf_dataframe()
loader = dataframe_to_torch_dataloader(df, target_column="target_family", batch_size=32)
```

Install optional PyTorch support manually:

```bash
pip install torch
```

## CNN Multitask Trainer

Train one CNN model that predicts both drone family and drone mode from raw H/L DroneRF signals.

Quick run:

```bash
uv run python cnn.py --epochs 3 --batch-size 16 --max-values-per-archive 50000 --device auto --auto-test
```

Full guide:

- [docs/cnn-trainer-guide.md](docs/cnn-trainer-guide.md)
