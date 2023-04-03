#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 03:32:26 2022

The class of glove solver to preprocess input

@author: Xinyang Chen
"""
from pathlib import Path
import os
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import cv2 as cv
import numpy as np
import copy

from glove_utils import HandDetector
from glove_utils.GlobalQueue import put_value, put_EOF

class GloveSolver():
    def __init__(self, lowBound=np.array([0, 127, 127]), highBound=np.array([255, 137, 137]),
                imgsz = (640, 640), buffer=np.array([0, 20, 20]), sampleSize=40, img=None, lab=None):
        # check definition of lowBound, highBound, sampleSize and segmentBuffer in MaskGenerator class
        self.imgsz = imgsz
        
        self.lowBound = lowBound # the low boundary of color mask
        self.highBound = highBound # the high boundary of color mask
        self.colorMask = np.zeros([self.imgsz[1], self.imgsz[0]], dtype = bool)
        
        self.sample_size = sampleSize # the size of each sample circle in fine sampling
        self.buffer = buffer # the shifting amount in LAB color space
        self.bound_state = False # the status of shifing amount (self.buffer[0]) initialization
        
        self.img = img # the image on present frame
        self.lab = lab # the image in LAB color space
        self.success_detect = False
    
    
    def getColorMask(self, lab):
        trim = cv.inRange(lab, self.lowBound, self.highBound)            
        check = np.zeros([trim.shape[0], trim.shape[1], 1])
        check[:,:,0]= trim[:,:]
        self.colorMask = np.all(check == 255,axis=-1)
        
    def colorChange(self, labSample, LArea, ABArea, amount):   
        """
        @brief: change color inside LAB color space with buffer
        param labSample: the image sample in LAB color space
        param area: the area whose A and B need to be changed
        param amount: the shifting amount in lab color space
        
        return labSample: the image sample after color shifting
        """
        labSample[LArea, 0] += amount[0]
        labSample[ABArea, 1] += amount[1]
        labSample[ABArea, 2] += amount[2]
        
        return labSample

    def bufferFineAdjust(self, landmarks, lab, mask):
        """
        @brief: readjust shifting amount of L by fine sampling
        param landmarks: the hand landmarks corresponding to lab
        param lab: the input of image in LAB color space
        param mask: a mask to calibrate the fine sampled area
        """
        sample_area = np.zeros([lab.shape[1], lab.shape[0]], dtype = bool)
        # generate some circle based on the position of hand landmarks
        for landmark in landmarks:
            temp = np.ones([lab.shape[1], lab.shape[0]], dtype = bool)
                    
            temp[int(landmark[1]) + self.sample_size:, :] = False
            temp[:int(landmark[1]) - self.sample_size, :] = False
            temp[:, int(landmark[2]) + self.sample_size:] = False
            temp[:, :int(landmark[2]) - self.sample_size] = False

            sample_area = sample_area | temp
        
        sample_area = mask & sample_area.transpose()
        if lab[sample_area, 0].shape[0] != 0:
            self.buffer[0] =  255 - np.max(lab[sample_area, 0])


    def bufferInitialize(self, ABArea, detector):
        """
        @brief: readjust shifting amount of L by fine sampling
        param area: the sample area for buffer initialization
        param detector: a mediapipe hand detector
        """
        MAX_UINT8 = 255
        stepSize = 5
        LArea = np.ones_like(ABArea)
        # generate a copy for present lab color
        labCopy = self.lab.copy()
        
        # initialize shifting amount of l with maximum inside candidate area
        self.buffer[0] =  MAX_UINT8 - np.max(labCopy[ABArea, 0])

        # run loop to search the lowest succeed shifting amount
        while True:
            newLab = self.colorChange(labCopy.copy(), LArea, ABArea, self.buffer)
            newImg = cv.cvtColor(newLab, cv.COLOR_LAB2BGR) 
            success = detector.findHands(newImg)
            tmpPositionList = detector.findPosition()
            if success:
                self.bufferFineAdjust(tmpPositionList, labCopy, ABArea)
                self.bound_state = True
                break
            if self.buffer[0] >= MAX_UINT8:
                # shift back when reach 255, fail and break
                break
            self.buffer[0] += stepSize
    
    def frameInitialize(self, img):
        """
        @brief: set img, blur the img and get lab by convert img into LAB color space
                also get the color mask based on threshold
        param img: the input image
        """
        self.success_detect = False
        self.img = img.copy()
        self.img = cv.resize(self.img, self.imgsz)
        self.lab = cv.cvtColor(self.img, cv.COLOR_BGR2LAB)
        self.getColorMask(self.lab)  
    
    def baseTrial(self, detector):
        self.getColorMask(self.lab) 
        LArea = np.ones_like(self.colorMask)
        newLab = self.colorChange(copy.deepcopy(self.lab), LArea, self.colorMask, self.buffer)
        newImg = cv.cvtColor(newLab, cv.COLOR_LAB2BGR) 
        self.success_detect = detector.findHands(newImg, newImg, True)
        self.img = newImg
    
    def solve(self, img, detector):
        """
        @brief: preprocess the image and trace the glove
        param img: input raw image
        param detector: a mediapipe hand detector
        
        return img: visualization result of glove solver
        """
        img_shape = (img.shape[1], img.shape[0])
        self.frameInitialize(img)
        
        # before initialization succeed, keep try initialization 
        if not self.bound_state:
            self.bufferInitialize(self.colorMask, detector)            
        
        # if succeed, preprocess the image and tracing the hand
        else:
            self.baseTrial(detector)  
            
        self.img = cv.resize(self.img, img_shape)
        
        
        
def run_baseline(low_bound=[0, 127, 127], 
        high_bound=[255, 137, 137], 
        imgsz=(640, 640), 
        source=0,
        nosave=False,
        save_dir=ROOT / 'runs/baseline',
        view_img=True,
        is_img=False):
    detector = HandDetector(imgsz[0], imgsz[1])
    solver = GloveSolver(np.array(low_bound), np.array(high_bound), imgsz)
    
    if str.isdigit(source):
        cap = cv.VideoCapture(int(source))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, imgsz[0])
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, imgsz[1])
    elif not is_img:
        cap = cv.VideoCapture(source)
    else:
        img = cv.imread(source)
        img = cv.resize(img, imgsz)   
        solver.solver(img, detector)
        if view_img:
            cv.imshow("result", solver.img)
            key = cv.waitKey(1)  # 1 millisecond
        return
    put_value([nosave, view_img, is_img])
    if not nosave:
        # Define the codec and create VideoWriter object
        _, name = os.path.split(source)
        name, _ = os.path.splitext(name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        put_value([str(Path(save_dir / name)), (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))])
        save_path = str(Path(str(Path(save_dir / name)) + "_track").with_suffix('.mp4'))  # force *.mp4 suffix on results videos
        print("Save Tracking Result to " + save_path)
        out = cv.VideoWriter(save_path,cv.VideoWriter_fourcc(*'mp4v'), 30.0, (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    

    success_read = True
    while success_read:  
        #camera check
        success_read, img = cap.read()
        if not success_read:
            if not nosave:
                out.release()
            break
      
        solver.solve(img, detector)
        
        put_value([copy.deepcopy(img), copy.deepcopy(detector.results)])
        if not nosave:
            out.write(solver.img)
        #cvdraw
        if str.isdigit(source) or view_img:
            cv.imshow("result", solver.img)
            key = cv.waitKey(1)  # 1 millisecond
            #terminate
            if key == 27 or key == ord('q'): 
                break 

    put_EOF()
    cap.release()