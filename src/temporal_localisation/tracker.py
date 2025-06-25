import cv2
import typing as t

from ultralytics import YOLO
from src.metrics import compute_iou
from src.kalman_filter import KalmanFilterBox

VEHICLE_CLASSES = {
    2: "car",
    7: "truck"
}


def find_bbox_vehicle_conf(
    frame,
    bbox: tuple,
    model: YOLO
) -> float:
    """
    Returns the maximum confidence that the bounding box
    for the given frame is a car, bus or truck.
    """
    x_min = max(0, bbox[0])
    y_min = max(0, bbox[1])
    x_max = min(frame.shape[1], bbox[0] + bbox[2])
    y_max = min(frame.shape[0], bbox[1] + bbox[3])

    if x_min >= x_max or y_min >= y_max:
        return 0.0

    vehicle = frame[y_min:y_max, x_min:x_max]
    output = model(vehicle, verbose=False)
    confidence = max([box.conf for box in output[0].boxes if box.cls in VEHICLE_CLASSES] + [0.0])

    return confidence

def find_car_bboxes(frame, model: YOLO) -> list:
    """
    Given a frame and the YOLO model, searches for the
    objects that match a: car, bus or a truck.
    """
    yolo_boxes = []
    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls)
        
        if cls in VEHICLE_CLASSES:
            # only take into consideration classes related
            # to vehicles: car, bus, truck
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            width, height = x_max - x_min, y_max - y_min
            yolo_boxes.append((x_min, y_min, width, height))

    return yolo_boxes

def find_best_match(csrt_bbox, yolo_boxes):
    """
    Finds the best match between the tracker box and the YOLO box.
    
    Useful in situations when the machine is making a turn, or the
    perspective is being changed, because its size and orientation
    will also be changing.
    """
    best_iou = 0
    best_box = None

    for yolo_box in yolo_boxes:
        iou = compute_iou(csrt_bbox, yolo_box)

        if iou > best_iou:
            best_iou = iou
            best_box = yolo_box
    
    return best_box, best_iou


def write_coordinates(
    coordinates: list[tuple],
    video: str,
    path: str,
) -> None:
    """
    Given a list of coordinates of the tracked car from a video,
    writes the coordinates in the file given in the `path`.
    """
    full_path = f"{path}{video}.txt"

    with open(full_path, "w") as f:
        f.write(f"{len(coordinates)} -1 -1 -1- 1\n")

        for i, bbox in enumerate(coordinates):
            f.write(f"{i} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")


def bbox_to_coordinates(bbox: tuple) -> tuple:
    """
    Transforms (x, y, width, height) to (x_min, y_min, x_max, y_max).
    """
    return (
        bbox[0],
        bbox[1],
        bbox[0] + bbox[2],
        bbox[1] + bbox[3],
    )


def track_object_with_yolo_validation(
    video_path: str,
    bbox: tuple[int],
    frames: t.Optional[int],
    reinitialization_rate: int = 5, 
    show: bool = False
) -> list[tuple]:
    tracker = cv2.TrackerKCF_create()
    capture = cv2.VideoCapture(video_path)
    model = YOLO("yolov8n.pt")
    frame_cnt = 0
    preds = [bbox_to_coordinates(bbox)]

    if not capture.isOpened():
        print(f"ERROR: Could not open video source {video_path}")
        return

    ok, frame = capture.read()

    if not ok:
        print("ERROR: Cannot read the first frame from video.")
        return

    # initialize the tracker
    ok = tracker.init(frame, bbox)
    # initialize the Kalman predictor
    kf = KalmanFilterBox(bbox)

    while True:
        ok, frame = capture.read()
        reinit_attempted = False
        current_bbox = None

        if not ok or (frames is not None and frame_cnt >= frames):
            # reached the final of the video,
            # by processing all the frames
            break

        # update the tracker
        ok, csrt_bbox = tracker.update(frame)
        # update the frame counter
        frame_cnt += 1
        # Kalman filter prediction
        kf_predicted_bbox = kf.predict()
        # base the inital prediction for the current frame
        # on the Kalman filter; if the CSRT tracker or the YOLO
        # produce a better prediction, update this
        current_bbox = kf_predicted_bbox

        if ok:
            csrt_bbox = tuple(map(int, csrt_bbox))
            iou_csrt_kf = compute_iou(csrt_bbox, kf_predicted_bbox)

            if iou_csrt_kf >= 0.3:
                # if the iou between CSRT tracker and Kalman is good enough,
                # the tracker did not drift and can be used for tracking
                current_bbox = csrt_bbox
                kf.update(csrt_bbox)
            else:
                current_bbox = kf_predicted_bbox

        # run YOLO periodically to reinitialize the tracking
        # object, because the motion can change its perspective
        if frame_cnt % reinitialization_rate == 0:
            # search for the best YOLO box 
            yolo_bboxes = find_car_bboxes(frame, model)
            best_match, best_iou = find_best_match(current_bbox, yolo_bboxes)

            if best_iou > 0.3:
                if best_match:
                    # reinitialize the tracked with the 
                    # bounding box computed by YOLO
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, best_match)

                    # update the current bbox
                    current_bbox = best_match
                    # update the Kalman bbox
                    kf.update(best_match)
                    reinit_attempted = True
                else:
                    reinit_attempted = False

        if show:
            # draw the current bounding box
            x, y, w, h = map(int, current_bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"Tracking" if ok and not reinit_attempted else "Reinitialized" if reinit_attempted else "Holding"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Hybrid CSRT + YOLO Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # transform the bounding box 
        # from (x, y, width, height) 
        # to   (x_min, y_min, x_max, y_max)
        # and add thit to the list with all predictions
        preds.append(bbox_to_coordinates(current_bbox))

    capture.release()
    
    if show:
        cv2.destroyAllWindows()

    return preds