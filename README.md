# Multi-Camera Vehicle Tracking and Surveillance

This project implements a video analysis pipeline for multi-camera surveillance of a road intersection, focusing specifically on **cross-view single vehicle tracking**.

## Task

Given video streams from multiple fixed cameras monitoring an intersection, the system's primary goal is to accurately track a single vehicle across these diverse viewpoints, accounting for changes in scale, orientation, and potential occlusions.

## Key Features & Methodologies

* **Object Detection and Tracking**:
    * Leverages real-time object detection models (e.g., YOLO) for initial vehicle identification.
    * Incorporates robust object trackers (e.g., CSRT/KCF) for continuous tracking within a single view.
    * Employs periodic re-initialization of the tracker using YOLO detections to maintain accuracy.
    * (Initially explored and later refined: Kalman Filters for motion prediction, though ultimately found to have limitations in handling severe occlusions with tracker-dependent updates.)

* **Cross-View Vehicle Identification (Homography-based)**:
    * Addresses the challenge of identifying the same vehicle across different camera angles.
    * **Homography Projections**: The core approach involves establishing a universal 2D coordinate system (e.g., a reference image from Google Earth).
        * Points/bounding boxes from one camera (e.g., Camera A) are projected onto this 2D reference plane using a pre-computed homography matrix.
        * These points are then projected from the 2D reference plane back into the coordinate system of another camera (e.g., Camera B) using an inverse homography.
    * (Initially explored and refined: Epipolar geometry was explored, where epipolar lines from Camera A were used to constrain the search space in Camera B for YOLO-detected vehicles. While a theoretically sound approach, practical challenges with sparse reference points led to its refinement in favor of homography for direct projection.)

* **Calibration Data Persistence**:
    * Homography matrices are pre-computed using manually selected corresponding points between camera views and the 2D reference.
    * These matrices are saved and loaded to avoid re-computation, ensuring efficient and consistent operation across multiple video processing runs.

# Environment

All the packages and their versions are included in the **requirements.txt** file. The recommended way to run this code is to use a **virtual environment**. After activating the virtual environment and installing all the dependencies, there are two variables that need to be taken into account when running the code (variables inside the main.py file):
- *CROSS_VIEW_TRACKING_INPUT_DIR*: where the videos for the *second task* will be read from
- *CROSS_VIEW_TRACKING_OUTPUT_DIR*: where the predicted bounding boxes will be stored

To actually run the code, the *main.py* file needs to be executed with the command *python main.py*.
