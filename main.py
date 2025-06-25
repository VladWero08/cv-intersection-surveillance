import cv2
import typing as t

from ultralytics import YOLO
from src.metrics import compute_iou, evaluate_tracking
from src.kalman_filter import KalmanFilterBox
from src.utils import get_camera_type_frame, get_fundamental_matrix, get_epipolar_line, draw_epipolar_line_on_frame

from src.temporal_localisation.tracker import write_coordinates, track_object_with_yolo_validation

TEMPORAL_LOCALISATION_INPUT_DIR = "./train/task2/"
TEMPORAL_LOCALISATION_OUTPUT_DIR = "./Olaeriu_Vlad_Mihai_407/task2/"
VIDEOS = ["01_1", "02_1", "03_1", "04_1", "05_1", "06_1", "07_1", "08_1", "09_1", "10_1", "11_1", "12_1", "13_1", "14_1", "15_1"]


def temporal_localisation() -> None:
    for VIDEO in VIDEOS:
        print(f"[INFO] Tracking Camera A car in {VIDEO}...")

        # format the path to the video
        video_path = f"{TEMPORAL_LOCALISATION_INPUT_DIR}{VIDEO}.mp4"
        # format the path to the bounding box
        bbox_path = f"{TEMPORAL_LOCALISATION_INPUT_DIR}{VIDEO}.txt"

        try:
            with open(bbox_path, "r+") as f:
                lines = f.readlines()
                # read the number of frames in the video 
                frames = int(lines[0].split()[0])
                # read the coordinates of the bounding box
                coords = list(map(int, lines[1].split()[1:]))

                # transform coordinates 
                # from (x_min, y_min, x_max, y_max)
                # to   (x_min, y_min, width, height)
                x, y = coords[0], coords[1]
                w, h = coords[2] - coords[0], coords[3] - coords[1]
                bbox = (x, y, w, h)
        except Exception as e:
            print(f"[ERROR]: While reading bounding box for {video_path}: {e}")
            continue

        # predict the coordinates for the car
        # given in the current video
        bboxes_preds_coords = track_object_with_yolo_validation(video_path, bbox, frames, show=False)
        print("[INFO] Finished tracking.")

        # write the predicted coordinates 
        write_coordinates(
            coordinates=bboxes_preds_coords,
            video=VIDEO,
            path=TEMPORAL_LOCALISATION_OUTPUT_DIR
        )
        print("[INFO] Finished writing results.")
        print()


if __name__ == "__main__":
    temporal_localisation()
    