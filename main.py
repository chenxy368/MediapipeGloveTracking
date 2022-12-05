#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 03:32:26 2022

The online test program of glove solver class

@author: Xinyang Chen
"""
from glove_utils import HandDetector
from glove_utils import GloveSolver

import cv2 as cv
import numpy as np
import time

def main(yolo, webcam=False):
    # fps
    pTime = time.time()
    cap_width = 640
    cap_height = 384
    #camera
    if webcam:
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    else:
        cap = cv.VideoCapture("sample.mp4")

        
    detector = HandDetector(cap_width, cap_height)
    low_bound = np.array([0, 127, 127])
    high_bound = np.array([255, 137, 137])
    imgsz = (cap_width, cap_height)
    solver = GloveSolver(low_bound, high_bound, imgsz)

    
    
    while True:  
        #camera check
        success_read, img = cap.read()
        if not success_read:
            break
        img = cv.resize(img, (cap_width, cap_height))        
        solver.solver(img, detector)

        #fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        cv.putText(solver.img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)        
        pTime = cTime 
    
        #cvdraw
        cv.imshow("result", solver.img)
        key = cv.waitKey(1)  # 1 millisecond
        
        #terminate
        if key == 27 or key == ord('q'): 
            break 

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    yolo = True
    main(yolo)
        