import numpy as np
import tensorflow as tf
import cv2
import os
import six.moves.urllib as urllib
import tarfile
from utils import label_map_util
from utils import visualization_utils as vis_util
from data_preparation import prepare_data

# Load the LSTM model
lstm_model = tf.keras.models.load_model('lstm_model_idd.h5')

# Object Detection Model Preparation
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

if not os.path.exists(MODEL_NAME):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    tar_file.extractall()

# Load TensorFlow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Helper function to load image into numpy array
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# Video input setup
cap = cv2.VideoCapture('input_video.mp4')  # Replace with your video file

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Extract objects and bounding boxes
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Visualization of the results of a detection
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            # Cut-in detection and collision warning logic
            for i, b in enumerate(boxes[0]):
                if classes[0][i] in [3, 6, 8]:  # Car, Bus, Truck classes
                    if scores[0][i] >= 0.5:
                        mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                        mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                        apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1])) ** 4), 1)
                        cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x * 800), int(mid_y * 450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        if apx_distance <= 0.5 and 0.3 < mid_x < 0.7:
                            cv2.putText(image_np, 'WARNING!!!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # Display output
            cv2.imshow('Detection', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
