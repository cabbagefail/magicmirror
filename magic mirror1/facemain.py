import cv2
import dlib
import numpy as np
from face_add_v1_2 import calculate_ear, apply_sticker, apply_whiskers




# 获取左眼中心点
def left_eye_center():
    left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    return np.mean(left_eye_points, axis=0).astype(int)


# 获取右眼中心点
def right_eye_center():
    right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
    return np.mean(right_eye_points, axis=0).astype(int)




# 获取嘴巴中心点
def mouth_center():
    mouth_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
    return np.mean(mouth_points, axis=0).astype(int)


# 计算嘴巴张开度（垂直距离/水平距离）
def calculate_mar(mouth_points):
    # 垂直距离：上唇下缘到下唇上缘
    vertical_dist = np.linalg.norm(np.array(mouth_points[13]) - np.array(mouth_points[19]))
    # 水平距离：嘴角左右距离
    horizontal_dist = np.linalg.norm(np.array(mouth_points[0]) - np.array(mouth_points[6]))
    return vertical_dist / horizontal_dist





