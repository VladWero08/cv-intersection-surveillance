def compute_iou(box_A: tuple[int], box_B: tuple[int]) -> float:
    """
    Given two bounding boxes, computes and returns the
    intersection under union.
    """
    x_A = max(box_A[0], box_B[0])
    y_A = max(box_A[1], box_B[1])
    x_B = min(box_A[0] + box_A[2], box_B[0] + box_B[2])
    y_B = min(box_A[1] + box_A[3], box_B[1] + box_B[3])

    intersection_area = max(0, x_B - x_A) * max(0, y_B - y_A)
    box_A_area = box_A[2] * box_A[3]
    box_B_area = box_B[2] * box_B[3]

    return intersection_area / float(box_A_area + box_B_area - intersection_area + 1e-5)

def evaluate_tracking(
    predictions: list[tuple], 
    ground_truth: list[tuple],
    iou_threshold: float = 0.3
) -> float:
    accuracy = 0

    for i in range(len(predictions)):
        preds_coords = predictions[i]
        gt_coords = ground_truth[i]

        if compute_iou(preds_coords, gt_coords) > iou_threshold:
            accuracy += 1

    return round(accuracy / len(predictions) * 100, 2)