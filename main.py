#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 03:32:26 2022

Main

@author: Xinyang Chen
"""
from glove_utils import run_baseline
from glove_utils import init
from yolov7_utils import run
from recognition_utils.app import run_recognition

import os
import sys
from pathlib import Path
from threading import Thread

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import argparse
import cv2

from yolov7_utils.utils.general import print_args, check_requirements

def parse_opt():
    solution_parser = argparse.ArgumentParser()
    solution_parser.add_argument('--baseline', action='store_true', help='run baseline or yolo_glove_tracker')
    solution_parser.add_argument('--norecognition', default=False, action='store_true', help='run recognition module')
    solution_parser.add_argument('--kalmanfilter', default=False, action='store_true', help='run kalman filter module')
    solution_parser.add_argument('--target_gesture', type=str, default='five', help='target gesture of dynamic recognition')
    solution_type, _ = solution_parser.parse_known_args()
    
    
    if solution_type.baseline:
        # baseline arg
        baseline_parser = argparse.ArgumentParser()
        baseline_parser.add_argument('--low_bound', nargs='+', type=int, help='low boundary for color mask in CIELAB colorspace')
        baseline_parser.add_argument('--high_bound', nargs='+', type=int, help='high boundary for color mask in CIELAB colorspace')
        baseline_parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
        baseline_parser.add_argument('--source', type=str, default='0', help='file, num for webcam')
        baseline_parser.add_argument('--nosave', default = False, action='store_true', help='do not save results')
        baseline_parser.add_argument('--save_dir', default = ROOT / 'runs/baseline', help='save results to dir')
        baseline_parser.add_argument('--view-img', default = False, action='store_true', help='show results')
        opt, _ = baseline_parser.parse_known_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        print_args(vars(opt))
        
        return solution_type, opt
    
    # yolo arg
    yolo_parser = argparse.ArgumentParser()
    yolo_parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
    yolo_parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    yolo_parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    yolo_parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    yolo_parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    yolo_parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    yolo_parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    yolo_parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    yolo_parser.add_argument('--view-img', action='store_true', help='show results')
    yolo_parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    yolo_parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    yolo_parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    yolo_parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    yolo_parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    yolo_parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    yolo_parser.add_argument('--augment', action='store_true', help='augmented inference')
    yolo_parser.add_argument('--visualize', action='store_true', help='visualize features')
    yolo_parser.add_argument('--update', action='store_true', help='update all models')
    yolo_parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    yolo_parser.add_argument('--name', default='exp', help='save results to project/name')
    yolo_parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    yolo_parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    yolo_parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    yolo_parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    yolo_parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    yolo_parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    yolo_parser.add_argument('--dataset_generation', default=False, action='store_true', help='use OpenCV for dataset generation')
    yolo_parser.add_argument('--color-shift', default=False, action='store_true', help='Using color shift for glove preprocessing')
    opt, _ = yolo_parser.parse_known_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return solution_type, opt


def main(solution_type, opt):
    init(solution_type)
    gesture = str(solution_type.target_gesture)
    print(gesture)
    
    if not solution_type.norecognition:
        t1 = Thread(target = run_recognition, args=(gesture,))
        t1.start()

    if solution_type.baseline:
        t2 = Thread(target = run_baseline(**vars(opt)))
        t2.start()
    else:
        check_requirements(exclude=('tensorboard', 'thop'))
        t2 = Thread(target = run(**vars(opt)))
        t2.start()

    cv2.destroyAllWindows()
    if not solution_type.norecognition:
        t1.join()
    t2.join()

if __name__ == "__main__":
    solution_type, opt = parse_opt()
    main(solution_type, opt)
        