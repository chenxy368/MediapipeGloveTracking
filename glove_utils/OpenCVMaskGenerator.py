#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 03:32:26 2022

The mask generator class

@author: Xinyang Chen
"""

import cv2 as cv
import numpy as np

from yolov7_util.yolov7mask import Yolov7Generator

class MaskGenerator():
    def __init__(self, lowBound, highBound, imgsz):
        self.lowBound = lowBound # the low boundary of color mask
        self.highBound = highBound # the high boundary of color mask
        self.imgsz = imgsz
        self.colorMask = np.zeros([self.imgsz[1], self.imgsz[0]], dtype = bool)
        
        self.yoloMask = np.zeros([self.imgsz[1], self.imgsz[0]], dtype = bool)
        self.yolov7 = Yolov7Generator(imgsz)

    def getColorMask(self, lab):
        trim = cv.inRange(lab, self.lowBound, self.highBound)            
        check = np.zeros([trim.shape[0], trim.shape[1], 1])
        check[:,:,0]= trim[:,:]
        self.colorMask = np.all(check == 255,axis=-1)

    def getYolov7Mask(self, image):
        mask = self.yolov7.run(image)
        cv.imshow("mask", mask)
        cv.waitKey(1)  # 1 millisecond
        check = np.zeros([mask.shape[0], mask.shape[1], 1])
        check[:, :, 0]= mask[:, :, 0]
        self.yoloMask = np.all(check != 0,axis=-1)