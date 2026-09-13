# Real-Time Vehicle Detection, Tracking, Counting,
# Speed Estimation and Traffic Density Analysis
# Using YOLO and Supervision

import cv2
import numpy as np
import csv
from datetime import datetime

import supervision as sv
from ultralytics import YOLO

from src import Annotator, ViewTransformer, SpeedEstimator
from src.traffic_density import calculate_density

from config import (
    IN_VIDEO_PATH,
    OUT_VIDEO_PATH,
    YOLO_MODEL_PATH,
    LINE_Y,
    SOURCE_POINTS,
    TARGET_POINTS,
    WINDOW_NAME,
)


# ============================================================
# 1. INITIALIZATION AND SETUP
# ============================================================

# Load YOLO model
model = YOLO(YOLO_MODEL_PATH)

# Get video information
video_info = sv.VideoInfo.from_video_path(IN_VIDEO_PATH)
print(video_info)

# Initialize ByteTrack tracker
tracker = sv.ByteTrack(frame_rate=video_info.fps)

# Define Line Zone for counting
offset = 55

start = sv.Point(offset, LINE_Y)
end = sv.Point(video_info.width - offset, LINE_Y)

line_zone = sv.LineZone(
    start,
    end,
    minimum_crossing_threshold=1
)


# ============================================================
# Perspective Transformation
# ============================================================

SOURCE = np.array(SOURCE_POINTS)
TARGET = np.array(TARGET_POINTS)

view_transformer = ViewTransformer(
    SOURCE,
    TARGET
)


# ============================================================
# Speed Estimator
# ============================================================

speed_estimator = SpeedEstimator(
    fps=video_info.fps,
    view_transformer=view_transformer
)


# ============================================================
# Annotator
# ============================================================

annotator = Annotator(
    resolution_wh=video_info.resolution_wh,
    box_annotator=True,
    label_annotator=True,
    line_annotator=True,
    multi_class_line_annotator=True,
    trace_annotator=True,
    polygon_zone=SOURCE,
)


# ============================================================
# OpenCV Window
# ============================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    video_info.width,
    video_info.height
)


# ============================================================
# 2. VIDEO PROCESSING
# ============================================================

frame_generator = sv.get_video_frames_generator(
    IN_VIDEO_PATH
)


# ============================================================
# CSV FILE FOR TRAFFIC ANALYTICS
# ============================================================

csv_file = open(
    "traffic_analysis.csv",
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "Time",
    "Vehicle_Count",
    "Traffic_Density",
    "Vehicles_In",
    "Vehicles_Out"
])


# ============================================================
# 3. MAIN LOOP
# ============================================================

with sv.VideoSink(
    OUT_VIDEO_PATH,
    video_info
) as sink:

    for frame_number, frame in enumerate(frame_generator):

        # ----------------------------------------------------
        # YOLO Detection
        # ----------------------------------------------------

        results = model(
            frame,
            verbose=False
        )[0]

        detections = sv.Detections.from_ultralytics(
            results
        )


        # ----------------------------------------------------
        # ByteTrack Tracking
        # ----------------------------------------------------

        detections = tracker.update_with_detections(
            detections
        )


        # ----------------------------------------------------
        # Line Zone Counting
        # ----------------------------------------------------

        line_zone.trigger(
            detections
        )


        # ----------------------------------------------------
        # Speed Estimation
        # ----------------------------------------------------

        detections = speed_estimator.update(
            detections
        )


        # ====================================================
        # NEW FEATURE 1:
        # CURRENT VEHICLE COUNT
        # ====================================================

        if detections.tracker_id is not None:
            vehicle_count = len(detections.tracker_id)
        else:
            vehicle_count = 0


        # ====================================================
        # NEW FEATURE 2:
        # TRAFFIC DENSITY
        # ====================================================

        traffic_density = calculate_density(
            vehicle_count
        )


        # ----------------------------------------------------
        # Create Labels
        # ID + Class + Speed
        # ----------------------------------------------------

        labels = []

        for tracker_id, class_name, speed in zip(
            detections.tracker_id,
            detections.data["class_name"],
            detections.data["speed"],
        ):

            text = f"{class_name} #{tracker_id}"

            if speed != 0:
                text = f"{class_name} {speed}km/h"

            labels.append(text)


        # ====================================================
        # Annotate Frame
        # ====================================================

        annotated_frame = annotator.annotate(
            frame,
            detections,
            labels=labels,
            line_zones=[line_zone],
            multi_class_zones=[line_zone],
        )


        # ====================================================
        # NEW FEATURE 3:
        # DISPLAY TRAFFIC INFORMATION
        # ====================================================

        cv2.rectangle(
            annotated_frame,
            (10, 10),
            (390, 125),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            annotated_frame,
            f"Vehicles: {vehicle_count}",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        cv2.putText(
            annotated_frame,
            f"Traffic: {traffic_density}",
            (25, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        cv2.putText(
            annotated_frame,
            f"In: {line_zone.in_count}  Out: {line_zone.out_count}",
            (25, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ====================================================
        # SAVE DATA TO CSV
        # ====================================================

        csv_writer.writerow([
            datetime.now().strftime("%H:%M:%S"),
            vehicle_count,
            traffic_density,
            line_zone.in_count,
            line_zone.out_count
        ])


        # ====================================================
        # WRITE OUTPUT VIDEO
        # ====================================================

        sink.write_frame(
            frame=annotated_frame
        )


        # ====================================================
        # SHOW REAL-TIME VIDEO
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            annotated_frame
        )


        # Press Q to stop
        if (
            cv2.waitKey(1) & 0xFF == ord("q")
            or
            cv2.getWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_VISIBLE
            ) < 1
        ):
            break


# ============================================================
# 4. CLEANUP
# ============================================================

csv_file.close()

cv2.destroyAllWindows()


# ============================================================
# 5. FINAL OUTPUT
# ============================================================

print("\nProcessing complete.")

print(
    f"Processed video saved at: {OUT_VIDEO_PATH}"
)

print(
    f"Traffic analysis saved at: traffic_analysis.csv"
)

print(
    f"Total vehicles counted: "
    f"{line_zone.in_count + line_zone.out_count}",
    end=" | "
)

print(
    f"(In: {line_zone.in_count}, "
    f"Out: {line_zone.out_count})"
)

print(
    f"Model used: {YOLO_MODEL_PATH}"
)