import cv2
import math
import numpy as np
from collections import deque
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort


# Initialize DeepSort for object tracking
def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE,
                        n_init=cfg_deep.DEEPSORT.N_INIT,
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)


# Function to estimate speed between two points
def estimatespeed(Location1, Location2):
    d_pixel = math.sqrt(math.pow(Location2[0] - Location1[0], 2) + math.pow(Location2[1] - Location1[1], 2))
    ppm = 8  # Pixels per meter
    d_meters = d_pixel / ppm
    time_constant = 15 * 3.6  # Time constant for speed calculation
    speed = d_meters * time_constant
    return int(speed)


# Function to check intersection of two line segments
def intersect(A, B, C, D):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


# Function to get direction based on movement
def get_direction(point1, point2):
    direction_str = ""
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"

    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"

    return direction_str


# Function to draw bounding boxes and handle events
def draw_boxes(img, bbox, names, object_id, identities=None):
    global data_deque, speed_line_queue, object_counter, object_counter1

    cv2.line(img, (200, 720), (500, 350), (255, 0, 0), 3)  # Line 1
    cv2.line(img, (1050, 720), (800, 350), (0, 0, 255), 3)  # Line 2

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]

        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))
        id = int(identities[i]) if identities is not None else 0
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
            speed_line_queue[id] = []

        # Add center to deque
        data_deque[id].appendleft(center)

        # Calculate speed and handle line crossing events
        if len(data_deque[id]) >= 2:
            direction = get_direction(data_deque[id][0], data_deque[id][1])
            object_speed = estimatespeed(data_deque[id][1], data_deque[id][0])
            speed_line_queue[id].append(object_speed)

            # Handle cut-in detection
            if intersect(data_deque[id][0], data_deque[id][1], (200, 720), (500, 350)):
                print(f"Cut-in detected on Line 1 by object {id}!")

            if intersect(data_deque[id][0], data_deque[id][1], (1050, 720), (800, 350)):
                print(f"Cut-in detected on Line 2 by object {id}!")

            # Handle collision warning
            if object_speed < 0:
                print(f"Collision warning for object {id}: Negative speed detected!")

        # Draw bounding box and annotations
        label = f'{id}:{names[object_id[i]]}'
        color = (0, 255, 0)  # Example color
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display counters or additional information on the frame
    # (Optional: Add your code for displaying counters)


# Main function to process video frames
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Example: Replace with your object detection and tracking function
        # Example code for YOLOv8 object detection and DeepSort tracking integration
        # Replace with your actual object detection and tracking implementation
        bbox = [[100, 100, 200, 200]]  # Example bounding box
        names = {0: 'Car'}  # Example class names
        object_id = [0]  # Example object ID
        identities = [1]  # Example identities

        draw_boxes(frame, bbox, names, object_id, identities)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    init_tracker()
    video_path = 'test3.mp4'  # Replace with your video path
    process_video(video_path)
