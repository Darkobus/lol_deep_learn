import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

model = YOLO(r".\TrainLoL\v1_base_model17\weights\best.pt")

monitor = {"top": 0, "left": 0, "width": 2560, "height": 1440}

with mss() as sct:
    while True:

        cv2.namedWindow("LOL", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("LOL", cv2.WND_PROP_TOPMOST, 1)
        cv2.resizeWindow("LOL", 960, 540)

        screenshot = sct.grab(monitor)

        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        results = model.predict(frame, conf=0.5, device=0, verbose=False)
        print(results)
        annotated_frame = results[0].plot()

        cv2.imshow("LOL", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
