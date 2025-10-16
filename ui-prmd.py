import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Subjects by side (column differs by side)
RIGHT = ["01", "02", "03", "04", "05", "06", "08", "09"]  # col 35
LEFT  = ["07", "10"]                                      # col 23

# Exercises to load
EXERCISES = ["m07", "m08", "m09", "m10"]

# Column map per exercise & side (adjust if m08 differs)
COLS = {
    "m07": {"right": 35, "left": 23},
    "m08": {"right": 34, "left": 22},
    "m09": {"right": 38, "left": 26},
    "m10": {"right": 35, "left": 23}
}

BASE = Path("data/ui-prmd")


def load_group(exercise: str, correctness: str, subjects: list[str], col_idx: int):
    """Load angles for a group of subjects for a given exercise and label."""
    rows = []
    for s in subjects:
        fname = (
            f"{exercise}_s{s}_angles.txt"
            if correctness == "Correct"
            else f"{exercise}_s{s}_angles_inc.txt"
        )
        path = BASE / correctness / "Kinect" / "Angles" / fname

        try:
            ang = np.genfromtxt(path)
            vals = ang[:, col_idx]
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        rows.extend({
            "subject": s,
            "side": "right" if s in RIGHT else "left",
            "exercise": exercise,
            "label": correctness,           # "Correct" or "Incorrect"
            "frame": i,
            "forearm_angle": float(v),
        } for i, v in enumerate(vals))
    return rows


def plot_forearm(df, exercise, subject, label):
    sub = df.query('exercise == @exercise and subject == @subject and label == @label').sort_values("frame")
    if sub.empty:
        print(f"No rows for {exercise}, s{subject}, {label}.")
        return
    plt.figure(figsize=(10,4))
    plt.plot(sub["frame"], sub["forearm_angle"])
    plt.title(f"Forearm Angle — {exercise} • s{subject} • {label}")
    plt.xlabel("Frame")
    plt.ylabel("Forearm angle (deg)")
    plt.tight_layout()
    plt.show()


def create_df():
    all_rows = []
    for ex in EXERCISES:
        all_rows += load_group(ex, "Correct", RIGHT,   COLS[ex]["right"])
        all_rows += load_group(ex, "Correct", LEFT,    COLS[ex]["left"])
        all_rows += load_group(ex, "Incorrect", RIGHT, COLS[ex]["right"])
        all_rows += load_group(ex, "Incorrect", LEFT,  COLS[ex]["left"])

    df = (pd.DataFrame(all_rows)
            .sort_values(["exercise", "subject", "label", "frame"])
            .reset_index(drop=True))

    for e in EXERCISES:
        for r in RIGHT:
            plot_forearm(df, e, r, "Incorrect")

    # Optional: save
    # df.to_csv("m07_m08_forearm_angles.csv", index=False)

    df.head(), df["exercise"].value_counts(), df.shape

    df.to_csv("ui-prmd_data.csv")

