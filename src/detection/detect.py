import torch
from ultralytics import YOLO

def basic_detection_train_lol_model(data_yaml, project_name, model_name="v1_base_model", device=0, plots=True, epochs=100, imgsz=640, batch=16):

    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    model = YOLO("yolo11s.pt")

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_name,
        name=model_name,
        plots=plots
    )

# TODO - test augmentation
#     noise=0.05,
#     blur=0.1,
#     hsv_s=0.3,
#     hsv_v=0.3,
#     perspective=0.0001,