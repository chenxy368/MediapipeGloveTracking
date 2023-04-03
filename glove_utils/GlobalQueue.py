# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 09:46:58 2023

@author: HP
"""
import queue


def init():
    global q
    q = queue.Queue(-1)


def put_value(val):
    q.put(val)


def get_value():
    return q.get()

def put_EOF():
    q.put([])
