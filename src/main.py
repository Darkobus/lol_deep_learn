from data_split import splitting_annotated_data_in_folder_train_and_val, splitting_video_into_frames, rename_current_labels_and_images_with_prefix
from detection.detect import basic_detection_train_lol_model

import os
from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOL_DATA_DIR = os.path.join(PROJECT_ROOT, "lol", "data")

DATA_DIR = os.path.join(LOL_DATA_DIR, "data")
DATA_YAML = os.path.join(LOL_DATA_DIR, "data.yaml")

# appended next and next annotated datasets
DATA1_DIR = os.path.join(LOL_DATA_DIR, "data1")
DATA2_DIR = os.path.join(LOL_DATA_DIR, "data2")
DATA3_DIR = os.path.join(LOL_DATA_DIR, "data3")


# rename_current_labels_and_images_with_prefix(DATA3_DIR,"_2")
splitting_annotated_data_in_folder_train_and_val(DATA3_DIR,DATA_DIR,"_x")


model = YOLO(r".\v1_base_model16\weights\best.pt")

results = model.predict(
    source=r".\League of Legends_02-15-2026_19-7-10-0.mp4",
    # source=r".\League of Legends_02-05-2026_23-16-0-0.mp4",
    save=True,
    conf=0.5,
    device=0
)

if __name__ == "__main__":
    basic_detection_train_lol_model(data_yaml=DATA_YAML,project_name="TrainLoL")

