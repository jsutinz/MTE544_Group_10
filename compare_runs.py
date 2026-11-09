import glob
import re
import numpy as np
from utilities import FileReader


def rough(x):
    # sample-to-sample jitter: lower = smoother
    return np.std(np.diff(x))


def lag_seconds(raw, filt, t):
    # delay of the filtered signal relative to the raw one (positive = filtered trails)
    dt = np.mean(np.diff(t))
    raw = raw - raw.mean()
    filt = filt - filt.mean()
    xc = np.correlate(filt, raw, mode="full")
    return (np.argmax(xc) - (len(raw) - 1)) * dt


rows = []
for fname in sorted(glob.glob("spiral_Q*_R*.csv")):
    m = re.search(r"Q([\d.]+)_R([\d.]+)\.csv", fname)
    q, r = float(m.group(1)), float(m.group(2))

    headers, values = FileReader(fname).read_file()
    d = np.array(values)
    col = lambda name: d[:, headers.index(name)]

    ax_ratio = rough(col("kf_ax")) / rough(col("imu_ax"))
    ay_ratio = rough(col("kf_ay")) / rough(col("imu_ay"))

    if "odom_v" in headers:
        t = col("stamp")
        v_lag = lag_seconds(col("odom_v"), col("kf_vx"), t)
        v_err = np.mean(np.abs(col("kf_vx") - col("odom_v")))
    else:
        v_lag, v_err = float("nan"), float("nan")

    rows.append((fname, q, r, ax_ratio, ay_ratio, v_lag, v_err))

print(f"{'file':28s} {'Q':>5s} {'R':>5s} {'ax ratio':>9s} {'ay ratio':>9s} {'v lag(s)':>9s} {'v err':>8s}")
for fname, q, r, axr, ayr, lag, err in rows:
    print(f"{fname:28s} {q:5.2f} {r:5.2f} {axr:9.3f} {ayr:9.3f} {lag:9.3f} {err:8.4f}")