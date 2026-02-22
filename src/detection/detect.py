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
        mosaic=1.0,
        mixup=0.2,
        hsv_v=0.6,
        perspective=0.0005,
        device=device,
        project=project_name,
        name=model_name,
        plots=plots
    )