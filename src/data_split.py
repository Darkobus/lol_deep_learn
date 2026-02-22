import splitfolders
import cv2
import os

import splitfolders


def rename_current_labels_and_images_with_prefix(folder_path, prefix="v2_"):
    img_path = os.path.join(folder_path, 'images')
    lbl_path = os.path.join(folder_path, 'labels')

    files = [file for file in os.listdir(img_path) if file.endswith(('.jpg', '.png', '.jpeg'))]

    for filename in files:
        old_name_no_ext = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]

        new_img_name = f"{prefix}{filename}"
        new_lbl_name = f"{prefix}{old_name_no_ext}.txt"

        os.rename(os.path.join(img_path, filename),
                  os.path.join(img_path, new_img_name))

        old_lbl_full = os.path.join(lbl_path, f"{old_name_no_ext}.txt")
        if os.path.exists(old_lbl_full):
            os.rename(old_lbl_full, os.path.join(lbl_path, new_lbl_name))


def splitting_annotated_data_in_folder_train_and_val(input_folder, output_folder, current_prefix):
    rename_current_labels_and_images_with_prefix(input_folder, prefix=current_prefix)

    splitfolders.ratio(input_folder, output=output_folder, seed=123, ratio=(.8, .2))

def splitting_video_into_frames(video_path,output_dir):
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