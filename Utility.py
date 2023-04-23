import pythoncom
pythoncom.CoInitialize()
import mediapipe as mp
import matplotlib
matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyrealsense2 as rs
import cv2
import numpy as np
from time import perf_counter
VISIBILITY_THRESHOLD = 0.5
WIDTH=640
HEIGHT=480
mode=0

WHITE = (0,0,0)

def caltime(func):
    def wrapper(*arg,**kwarg):
        t0=perf_counter()
        result = func(*arg,**kwarg)
        print(func.__name__ ,"%.5f s"%(perf_counter()-t0))
        return result
    return wrapper
