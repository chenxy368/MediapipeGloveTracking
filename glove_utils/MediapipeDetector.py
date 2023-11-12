#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 03:32:26 2022

The class of hand detector based on mediapipe

@author: Xinyang Chen
"""

import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib
import math
import cv2 as cv
import numpy as np

class HandDetector():
    def __init__(self, wid=640, hei=360, mode=False, maxHands=1, model_complexity=1, detectionCon=0.15, trackCon=0.3):
        self.mode = mode# hand tracking mode
        self.maxHands = maxHands# number of hands
        self.model_complexity = model_complexity# complexity of model
        self.detectionCon = detectionCon# detection confidence(palm detector)
        self.trackCon = trackCon# tracking confidence(landmark model)

        self.mpHands = mp.solutions.hands# hand tracking solution
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils# draw
        self.wid = wid # width of the input image
        self.hei = hei # height of the input image
        self.results = None

    def findHands(self, img, drawOn=None, draw=False):
        """
        @brief: get the hand tracing based on mediapipe
        param img: the input image
        param drawOn: draw tracing result on drawOn
        param draw: enable drawing

        return success: the status of tracing
        return drawOn: output after drawing
        """
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        success = False
        mp_drawing = mp.solutions.drawing_utils
        drawing_spec_lm = mp_drawing.DrawingSpec(thickness=-1, circle_radius=20, color=(255, 0, 0))
        drawing_spec_bone = mp_drawing.DrawingSpec(thickness=10, circle_radius=20, color=(0, 0, 0))

        if self.results.multi_hand_landmarks:
            success = True
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(drawOn, handLms, self.mpHands.HAND_CONNECTIONS, 
                                               landmark_drawing_spec=drawing_spec_lm, 
                                            connection_drawing_spec=drawing_spec_bone)
                    #self.plot_landmarks(handLms, self.mpHands.HAND_CONNECTIONS, 
                    #                           landmark_drawing_spec=drawing_spec_lm, 
                    #                        connection_drawing_spec=drawing_spec_bone)

        return success
    
    def plot_landmarks(self, landmark_list, connections, landmark_drawing_spec,
                   connection_drawing_spec, elevation: int = 10, azimuth: int = 20):
        matplotlib.use('Agg')
        _PRESENCE_THRESHOLD = 0.5
        _VISIBILITY_THRESHOLD = 0.5
        def _normalize_color(color):
            return tuple(v / 255. for v in color)

        if not landmark_list:
            return
        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        ax.view_init(elev=elevation, azim=azimuth)
        plotted_landmarks = {}
        for idx, landmark in enumerate(landmark_list.landmark):
            if ((landmark.HasField('visibility') and
                landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                landmark.presence < _PRESENCE_THRESHOLD)):
                continue
            ax.scatter3D(
                xs=[-landmark.z],
                ys=[landmark.x],
                zs=[-landmark.y],
                color=_normalize_color(landmark_drawing_spec.color[::-1]),
            linewidth=landmark_drawing_spec.thickness)
            plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
        if connections:
            num_landmarks = len(landmark_list.landmark)
            # Draws the connections if the start and end landmarks are both visible.
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                    raise ValueError(f'Landmark index is out of range. Invalid connection '
                         f'from landmark #{start_idx} to landmark #{end_idx}.')
                if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                    landmark_pair = [
                        plotted_landmarks[start_idx], plotted_landmarks[end_idx]
                    ]
                    ax.plot3D(
                        xs=[landmark_pair[0][0], landmark_pair[1][0]],
                        ys=[landmark_pair[0][1], landmark_pair[1][1]],
                        zs=[landmark_pair[0][2], landmark_pair[1][2]],
                        color=_normalize_color(connection_drawing_spec.color[::-1]),
                        linewidth=connection_drawing_spec.thickness)
        plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0)

    def boundingBox(self, wid, hei): 
        boundingBox = [self.wid, self.hei, 0, 0] # list of the landmarks
    
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handlms.landmark):
                    # x and y are normalized to [0.0, 1.0] by the image width and height respectively. 
                    curr_x = lm.x * wid
                    curr_y = lm.y * hei
                    boundingBox[0] = min(curr_x, boundingBox[0])
                    boundingBox[1] = min(curr_y, boundingBox[1])
                    boundingBox[2] = max(curr_x, boundingBox[2])
                    boundingBox[3] = max(curr_y, boundingBox[3])

        return boundingBox
            
    def findPosition(self, wid = -1, hei = -1):
        """
        @brief: get the hand tracing based on mediapipe
        param wid: the width of the input image
        param hei: the height of the input image
        return lmList: the list of hand landmarks, elements are id, x position and y position
        """
    
        lmList = [] # list of the landmarks
    
        if wid < 0:
            wid = self.wid
        if hei < 0:
            hei = self.hei
    
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handlms.landmark):
                    # x and y are normalized to [0.0, 1.0] by the image width and height respectively. 
                    lmList.append([id, lm.x * wid, lm.y * hei])

                
        return lmList
    
    def getBaseMask(self, wid = -1, hei = -1):
        mask = np.zeros([hei, wid], dtype=np.uint8)
        lmlist = self.findPosition(wid, hei)
        if lmlist == []:
            return mask

        polyList = np.array([[lmlist[0][1], lmlist[0][2]], \
                             [lmlist[1][1], lmlist[1][2]], \
                             [lmlist[2][1], lmlist[2][2]], \
                             [lmlist[5][1], lmlist[5][2]],\
                             [lmlist[9][1], lmlist[9][2]], \
                             [lmlist[13][1], lmlist[13][2]], \
                             [lmlist[17][1], lmlist[17][2]]], dtype = np.int)
        cv.fillPoly(mask, pts = [polyList], color = 255)


        for i in range(3):
            for j in range(5):
                startPointID = 4 * (j + 1) - i
                endPointID = startPointID - 1
                thickness = 10
                if j == 0:
                    thickness = 13
                elif j == 4:
                    thickness = 3
                cv.line(mask, (int(lmlist[startPointID][1]), int(lmlist[startPointID][2])), 
                    (int(lmlist[endPointID][1]), int(lmlist[endPointID][2])), 255, thickness)

        return mask

    def getSampleMask(self, wid = -1, hei = -1, max_r = 9):
        mask = np.zeros([hei, wid], dtype=np.uint8)
        lmlist = self.findPosition(wid, hei)
        if lmlist == []:
            return mask
        
        center_radius_list = []

        type1_list = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
        type2_list = [4, 8, 12 ,16]
        for lm_id in range(len(lmlist)):
            lm = lmlist[lm_id]
            if lm_id != 0:
                prev_lm = lmlist[lm_id-1] 
            if lm_id != len(lmlist) - 1:
                next_lm = lmlist[lm_id+1]

            if lm[0] == 0:
                continue

            if lm[0] == len(lmlist) - 1:
                center = (int(lm[1]), int(lm[2]))
                radius = 0.5 * math.sqrt((prev_lm[1] - center[0]) ** 2 + (prev_lm[2] - center[1]) ** 2)
                center_radius_list.append([center, radius])

            if lm[0] in type1_list:
                center = (int(lm[1]), int(lm[2]))
                radius = 0.25 * (math.sqrt((next_lm[1] - center[0]) ** 2 + (next_lm[2] - center[1]) ** 2) \
                                + (math.sqrt((prev_lm[1] - center[0]) ** 2 + (prev_lm[2] - center[1]) ** 2)))
                center_radius_list.append([center, radius])
                continue

            if lm[0] in type2_list:
                center = (int(lm[1]), int(lm[2]))
                radius = 0.5 * math.sqrt((prev_lm[1] - center[0]) ** 2 + (prev_lm[2] - center[1]) ** 2)
                center_radius_list.append([center, radius])

                lm = lmlist[0]
                one_third = [int((2 * lm[1] + next_lm[1]) / 3), int((2 * lm[2] + next_lm[2]) / 3)]
                two_third = [int((lm[1] + 2 * next_lm[1]) / 3), int((lm[2] + 2 * next_lm[2]) / 3)]

                center = (int(one_third[0]), int(one_third[1]))
                radius = 0.25 * (math.sqrt((lm[1] - one_third[0]) ** 2 + (lm[2] - one_third[1]) ** 2) \
                                + (math.sqrt((one_third[0] - two_third[0]) ** 2 + (one_third[1] - two_third[1]) ** 2)))
                center_radius_list.append([center, radius])

                center = (int(two_third[0]), int(two_third[1]))
                radius = 0.25 * (math.sqrt((next_lm[1] - two_third[0]) ** 2 + (next_lm[2] - two_third[1]) ** 2) \
                                + (math.sqrt((one_third[0] - two_third[0]) ** 2 + (one_third[1] - two_third[1]) ** 2)))
                center_radius_list.append([center, radius])

        for circle in center_radius_list:
            center = circle[0]
            radius = circle[1]
            cv.circle(mask, center, int(radius), 255, -1)

        return mask

def findPosition(results):
    """
    @brief: get the hand tracing based on mediapipe
    param wid: the width of the input image
    param hei: the height of the input image

    return lmList: the list of hand landmarks, elements are id, x position and y position
    """
        
    lmList = np.array([], np.float32) # list of the landmarks
        
        
    if results.multi_hand_landmarks:
        lmList = np.zeros([21, 3], np.float32)
        for handlms in results.multi_hand_landmarks:
            for curr_id, lm in enumerate(handlms.landmark):
                # x and y are normalized to [0.0, 1.0] by the image width and height respectively. 
                lmList[curr_id, :] = np.array([lm.x, lm.y, lm.z], np.float32)
                    
    return lmList

def setPosition(lmList, results):
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for curr_id, lm in enumerate(handlms.landmark):
                # x and y are normalized to [0.0, 1.0] by the image width and height respectively. 
                lm.x = lmList[curr_id, 0]
                lm.y = lmList[curr_id, 1]
                lm.z = lmList[curr_id, 2]
                    
    return results