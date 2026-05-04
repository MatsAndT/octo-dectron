import csv
import matplotlib.pyplot as plt
import numpy as np

sample_rate = 4e7

def plot_csv_row(csv_path: str) -> None:
    """Read a single-row CSV and plot its values in time and frequency domains using matplotlib.pyplot."""
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        row = next(reader, None)

    if row is None:
        raise ValueError(f"CSV file '{csv_path}' is empty.")

    try:
        values = [float(value) for value in row]
    except ValueError as exc:
        raise ValueError("All values in the CSV row must be numeric.") from exc

    x = list(range(len(values)))
    t = [i / sample_rate for i in x]

    # Compute FFT
    fft_vals = np.fft.fft(values)
    freqs = np.fft.fftfreq(len(values), d=1/sample_rate)

    # Plot time domain
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(t, values, linewidth=0.5, color="blue")
    plt.title(f"Time Domain - Values from {csv_path}")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Plot frequency domain (positive frequencies only)
    plt.subplot(2, 1, 2)
    n = len(freqs) // 2
    plt.plot(freqs[:n] / 1e6, np.abs(fft_vals)[:n], linewidth=0.5, color="red")
    plt.title(f"Frequency Domain - Magnitude Spectrum from {csv_path}")
    plt.xlabel("Relative Frequency (MHz)")
    plt.ylabel("Magnitude")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_csv_row("11000L_0.csv")
