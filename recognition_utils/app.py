#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy

from pathlib import Path
import os
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from collections import Counter
from collections import deque

import cv2 as cv
from recognition_utils.utils import hand_gesture_mediapipe
from recognition_utils.utils import CvFpsCalc
from recognition_utils.utils.plot import (draw_landmarks, draw_bounding_rect, draw_info_text, draw_point_history, draw_info)
from recognition_utils.utils.data import (calc_bounding_rect, calc_landmark_list, pre_process_landmark, pre_process_point_history, logging_csv)
from recognition_utils.utils.scene import select_mode
from recognition_utils.model import KeyPointClassifier
from recognition_utils.model import PointHistoryClassifier

from glove_utils.GlobalQueue import get_value

# NUM_OF_FRAMES_PER_PRED = 30 # number of frames to make a gesture prediction
NUM_OF_FRAMES_PER_PRED = 45 # number of frames to make a gesture prediction
TEST_MODE = True # True: test model, False: collect data only
COLLECT_DATA = True # True: store the prediction results, False: don't store (real application)
'''
This can be used in video dataset creation!
'''
DATA_FILENAME = "model/point_history_classifier/datasets/prediction_results.pkl" # The filename to store the prediction results and key points (if COLLECT_DATA=True)

'''
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--video", type=str, default='')
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--store_result', default=COLLECT_DATA, action='store_true',
                        help="store prediction results")
    parser.add_argument("--store_file", default=DATA_FILENAME, type=str, help="output file containing stored prediction results")

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.75)

    args = parser.parse_args()

    return args
'''

def run_recognition():
    nosave, view_img, is_image = get_value()
    # Model load #############################################################
    keypoint_classifier = KeyPointClassifier()

    if TEST_MODE:
        """Single Model"""
        point_history_classifier = PointHistoryClassifier("recognition_utils/model/point_history_classifier/point_history_classifier_LSTM_ConquerCross.tflite")

    # Read labels ###########################################################
    with open('recognition_utils/model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [
            row[0] for row in csv.reader(f)
        ]
    with open(
            'recognition_utils/model/point_history_classifier/datasets/point_history_classifier_label.csv',
            # 'model/point_history_classifier/point_history_classifier_label_simple3.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = [
            row[0] for row in csv.reader(f)
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=20)

    # Coordinate history #################################################################
    history_length = NUM_OF_FRAMES_PER_PRED
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0


    # variable to store the processing history
    prediction_rst_dict = {
        "keypoint_pos": [],
        "prediction_label": point_history_classifier_labels,
        "prediction_rsts": [],
        "prediction_rsts_three_labels": [],
        "prediction_rsts_final": []
    }
    

    out = None

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        producer_val = get_value()
        if producer_val == []:
            if not nosave:
                out.release()
            break
        elif isinstance(producer_val[0], str):
            if isinstance(out, cv.VideoWriter):
                out.release()
            if not nosave:
                # Define the codec and create VideoWriter object
                save_path = producer_val[0]
                save_path = str(Path(save_path + "_recognize").with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                print("Save Recognition Result to " + save_path)
                out = cv.VideoWriter(save_path,cv.VideoWriter_fourcc(*'mp4v'), 30.0, producer_val[1])
            producer_val = get_value()
            
        image = producer_val[0]
        results = producer_val[1]
        debug_image = copy.deepcopy(image)

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # add static angle algo here
                debug_image, static_gesture = hand_gesture_mediapipe.detectWithDynamic(debug_image, hand_landmarks)
                if static_gesture == 'point':
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(
                        debug_image, point_history)
                    # Write to the dataset file
                    logging_csv(number, mode, pre_processed_landmark_list,
                                pre_processed_point_history_list)

                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    if hand_sign_id == 2:  # Point gesture
                        point_history.append(landmark_list[8])
                    else:
                        if point_history:
                            point_history.append(point_history[-1])
                        else:
                            point_history.append([0, 0])
                    
                    # store the keypoint_position for performance analysis
                    prediction_rst_dict["keypoint_pos"].append(point_history[-1])

                    # Finger gesture classification
                    finger_gesture_id = len(point_history_classifier_labels) - 1

                    point_history_len = len(pre_processed_point_history_list)
                    if TEST_MODE and point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(
                            pre_processed_point_history_list)

                    prediction_rst_dict["prediction_rsts"].append(finger_gesture_id)
                    prediction_rst_dict["prediction_rsts_three_labels"].append(finger_gesture_id)

                    # Calculates the gesture IDs in the latest detection
                    finger_gesture_history.append(finger_gesture_id)
                    
                    finger_gesture_id_counter = Counter(finger_gesture_history)
                    most_common_fg_id = finger_gesture_id_counter.most_common()

                    # prediction from most_common
                    finger_id_prediction_final = most_common_fg_id[0][0]
                    
                    prediction_rst_dict["prediction_rsts_final"].append(finger_id_prediction_final)
                  
                    # Drawing part 
                    debug_image = draw_bounding_rect(True, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    # only display the most commonly seen gesture in the last 
                    # NUM_OF_FRAMES_PER_PRED frames
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        keypoint_classifier_labels[hand_sign_id],
                        point_history_classifier_labels[finger_id_prediction_final],
                    )

                    debug_image = draw_point_history(debug_image, point_history)
        else:
            point_history.append([0, 0])

        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        if view_img:
            cv.imshow('Hand Gesture Recognition', debug_image)
        if not nosave:
            out.write(debug_image)

    # store the prediction result
    '''
    if args.store_result:
        print("prediction results are stored to:", args.store_file)
        with open(args.store_file, 'wb') as f:
            pkl.dump(prediction_rst_dict, f, pkl.HIGHEST_PROTOCOL)
    '''
