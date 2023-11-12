# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 09:46:58 2023

@author: HP
"""
import queue

from glove_utils.HandData import HandData
from glove_utils.MediapipeDetector import (findPosition, setPosition)

init_count = 30
arg_map = {}

def init(solution_type):
    global q
    q = queue.Queue(-1)
    global kalmanfilter_processor
    kalmanfilter_processor = HandData()
    global prev_result
    prev_result = None

    global arg_map
    arg_map = {"recognition": not solution_type.norecognition, 
               "kalmanfilter": solution_type.kalmanfilter}

def get_arg(key):
    return arg_map[key]

def set_arg(key, value):
    arg_map[key] = value

def kalmanfilter_process(val):
    global init_count
    global prev_result
    if len(val) == 2 and not isinstance(val[0], str):
        if val[1].multi_hand_landmarks: 
            kalmanfilter_processor.process(findPosition(val[1]))
            if init_count == 0:
                val[1] = setPosition(kalmanfilter_processor.currFramePos, val[1])
            elif init_count > 0:
                init_count -= 1
            prev_result = val[1]
        else:
            if prev_result:
                val[1] = prev_result
    elif len(val) == 2 and isinstance(val[0], str):
        kalmanfilter_processor.reset()
        init_count = 30

    return val

def put_value(val):
    global arg_map
    if arg_map["kalmanfilter"]:
        val = kalmanfilter_process(val)
    q.put(val)


def get_value():
    return q.get()

def put_EOF():
    q.put([])
