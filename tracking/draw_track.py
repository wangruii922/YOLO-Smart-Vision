import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs(
    "outputs/images",
    exist_ok=True
)

csv_path = (
    "outputs/csv/tracking_xy.csv"
)

df = pd.read_csv(
    csv_path
)

plt.figure(
    figsize=(14, 10)
)

track_ids = sorted(
    df["track_id"].unique()
)

scatter = None

for track_id in track_ids:

    target = df[
        df["track_id"]
        == track_id
    ]

    if len(target) < 3:
        continue

    x = target["x"].values

    y = target["y"].values

    frames = (
        target["frame"]
        .values
    )

    plt.plot(
        x,
        y,
        linewidth=1
    )

    scatter = plt.scatter(
        x,
        y,
        c=frames,
        s=20
    )

    plt.text(
        x[-1],
        y[-1],
        str(track_id),
        fontsize=8
    )

plt.gca().invert_yaxis()

plt.title(
    "Multi-Object Tracking Trajectories",
    fontsize=16
)

plt.xlabel(
    "X Position"
)

plt.ylabel(
    "Y Position"
)

plt.grid(True)

if scatter is not None:

    plt.colorbar(
        scatter,
        label="Frame Number"
    )

plt.tight_layout()

save_path = (
    "outputs/images/all_tracks.png"
)

plt.savefig(
    save_path,
    dpi=300
)

plt.show()

print(
    "\n轨迹图已保存:"
)

print(save_path)