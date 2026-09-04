from pathlib import Path
import cv2
import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATASET_PATH = Path(r"D:\Amrita SLR Dataset\sign project")

OUTPUT_FILE = Path("data/video_metadata.csv")


# --------------------------------------------------
# FIND VIDEOS
# --------------------------------------------------

records = []

for class_folder in sorted(DATASET_PATH.iterdir()):

    if not class_folder.is_dir():
        continue

    folder_name = class_folder.name

    # Separate numeric ID and label
    parts = folder_name.split(".", 1)

    if len(parts) == 2:
        class_id = int(parts[0])
        label = parts[1]
    else:
        class_id = None
        label = folder_name

    videos = list(class_folder.glob("*.mp4"))

    for video_path in videos:

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Could not open: {video_path}")
            continue

        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        duration = (
            frame_count / fps
            if fps > 0
            else 0
        )

        cap.release()

        records.append({
            "class_id": class_id,
            "label": label,
            "video_name": video_path.name,
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_seconds": duration
        })


# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(records)

df = df.sort_values(
    ["class_id", "video_name"]
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n========== DATASET SUMMARY ==========")

print(f"Total videos: {len(df)}")
print(f"Total classes: {df['class_id'].nunique()}")

print("\nVideos per class:")

print(
    df.groupby(
        ["class_id", "label"]
    ).size()
)

print("\nVideo properties:")

print(
    df[
        [
            "frame_count",
            "fps",
            "width",
            "height",
            "duration_seconds"
        ]
    ].describe()
)

print(
    f"\nMetadata saved to: {OUTPUT_FILE}"
)