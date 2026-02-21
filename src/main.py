from data_split import splitting_annotated_data_in_folder, splitting_video_into_slides
from detection.detect import basic_detection_train_lol_model
import os

DATA_DIR = r"C:\Users\Admin\PycharmProjects\lol_deep_learn\lol\data"
DATA_OUT_DIR = r"C:\Users\Admin\PycharmProjects\lol_deep_learn\lol\data\data"
VIDEO_PATH = r"C:\Users\Admin\PycharmProjects\lol_deep_learn\lol\data\League of Legends_02-05-2026_23-19-23-0.mp4"
VIDEO_OUTPUT_PATH = r"C:\Users\Admin\PycharmProjects\lol_deep_learn\lol\data\slides"


# splitting_annotated_data_in_folder(DATA_DIR,DATA_OUT_DIR)

# splitting_video_into_slides(VIDEO_PATH,VIDEO_OUTPUT_PATH)

# splitting_annotated_data_in_folder(VIDEO_OUTPUT_PATH,DATA_OUT_DIR)


from ultralytics import YOLO


model = YOLO(r"C:\Users\Admin\PycharmProjects\lol_deep_learn\runs\detect\TrainLoL\v1_base_model8\weights\best.pt")

results = model.predict(
    source=r"C:\Users\Admin\PycharmProjects\lol_deep_learn\videos\League of Legends_02-05-2026_23-2-23-0.mp4",
    save=True,
    conf=0.5,
    device=0
)

# if __name__ == "__main__":
#     DATA_YAML = r"C:\Users\Admin\PycharmProjects\lol_deep_learn\lol\data\data.yaml"
#     basic_detection_train_lol_model(data_yaml=DATA_YAML,project_name="TrainLoL")


