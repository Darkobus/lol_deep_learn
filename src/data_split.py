import splitfolders
import cv2
import os

import splitfolders


def splitting_annotated_data_in_folder(input_folder, output_folder):
    splitfolders.ratio(input_folder, output=output_folder,seed=123, ratio=(.8, .2), group_prefix=None)


def splitting_video_into_slides(video_path,output_dir):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = f"frame_{frame_count:06d}.jpg"
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)
        frame_count += 1

    cap.release()