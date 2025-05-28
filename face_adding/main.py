import cv2
import dlib
import numpy as np
from face_add_v1_2 import calculate_ear, apply_sticker, apply_whiskers

# 初始化dlib的人脸检测器和关键点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")  # 需下载模型文件

'''
准备贴纸
'''
# 加载眼镜图像（带alpha通道）
glass_img = cv2.imread("images/glass.png", cv2.IMREAD_UNCHANGED)
glass_2_img = cv2.imread("images/glass_2.png", cv2.IMREAD_UNCHANGED)
glass_3_img = cv2.imread("images/glass_3.png", cv2.IMREAD_UNCHANGED)

# 加载猫咪胡须
whiskers_img = cv2.imread("images/cat_1.png", cv2.IMREAD_UNCHANGED)

# 创建贴纸列表
sticker_list = [glass_img, glass_2_img, glass_3_img]

# 贴纸对应缩放系数
scale_factors = [0.45, 0.48, 0.33]

# 当前贴纸索引
current_sticker_index = 0

'''
眨眼切换特效
'''
# 眨眼检测参数
EAR_THRESHOLD = 0.2  # EAR 阈值，小于该值认为眼睛闭合
EAR_CONSEC_FRAMES = 4  # 连续几帧 EAR 小于阈值视为一次眨眼
blink_counter = 0  # 眨眼帧计数器
blink_history = []  # 存储眨眼的时间戳


# 获取左眼中心点
def left_eye_center():
    left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    return np.mean(left_eye_points, axis=0).astype(int)


# 获取右眼中心点
def right_eye_center():
    right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
    return np.mean(right_eye_points, axis=0).astype(int)


'''
微笑检测参数
'''
MAR_THRESHOLD = 0.25  # 调整此阈值（建议0.2-0.5）
MAR_CONSEC_FRAMES = 5
smile_counter = 0
timeout_threshold = 3  # 超时时间设为3秒
last_smile_time = 0  # 用于记录最后一次检测到微笑的时


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


# 设置视频捕获
cap = cv2.VideoCapture(0)

# 显示调试信息开关
de_show = True

show_whiskers = False  # 初始不显示胡须

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为灰度图像用于人脸检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = detector(gray, 0)

    for face in faces:
        # 关键点检测
        landmarks = predictor(gray, face)
        # 添加贴纸
        apply_sticker(frame, landmarks, current_sticker_index, sticker_list, scale_factors)

        # 获取嘴巴关键点
        mouth_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
        mar = calculate_mar(mouth_points)

        # 微笑检测
        if mar > MAR_THRESHOLD:
            smile_counter += 1
            if smile_counter >= MAR_CONSEC_FRAMES:
                show_whiskers = True  # 触发胡须特效
                last_smile_time = cv2.getTickCount()
                print("检测到微笑！已激活猫咪胡须")
                smile_counter = 0  # 重置计数器
        else:
            # 超时关闭特效
            smile_counter = 0
            # 检查是否超时
            if show_whiskers:
                time_since_last_smile = (cv2.getTickCount() - last_smile_time) / cv2.getTickFrequency()  # 除以频率，得到秒数
                if time_since_last_smile > timeout_threshold:
                    show_whiskers = False  # 关闭胡须特效
                    print("超时未微笑，已关闭猫咪胡须")

        # 新增猫胡须特效（仅当某个条件满足时启用，例如按键切换）
        if show_whiskers:  # show_whiskers 是一个布尔变量，可通过按键控制开关
            apply_whiskers(frame, landmarks, whiskers_img, scale_factor=2.3)

        # 获取眼部关键点坐标
        left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        # 计算左右眼的 EAR
        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0

        # 判断是否闭眼
        if avg_ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= EAR_CONSEC_FRAMES:
                # 检测到一次有效眨眼
                blink_history.append(cv2.getTickCount())
                # 保留最近两次眨眼的时间戳
                if len(blink_history) > 2:
                    blink_history.pop(0)

                # 检查是否为连续两次眨眼
                if len(blink_history) == 2:
                    time_diff = (blink_history[1] - blink_history[0]) / cv2.getTickFrequency()
                    if time_diff < 0.75:  # 两次眨眼间隔小于 0.75 秒
                        # 切换贴纸
                        current_sticker_index = (current_sticker_index + 1) % len(sticker_list)
                        print(f"切换贴纸至索引 {current_sticker_index}")
            blink_counter = 0

        # 绘制调试信息
        if de_show:

            # 绘制眼部关键点（共12个点）
            for point in left_eye_points + right_eye_points:
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)  # 绿色小圆点

            # 绘制眼睛中心点
            cv2.circle(frame, tuple(left_eye_center()), 3, (255, 0, 0), -1)  # 蓝色圆点
            cv2.circle(frame, tuple(right_eye_center()), 3, (255, 0, 0), -1)

            # 绘制眼镜中心线（连接两眼中心）
            cv2.line(frame, tuple(left_eye_center()), tuple(right_eye_center()), (0, 0, 255), 1)

            cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            #  绘制EAR
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 绘制嘴巴关键点
            for point in mouth_points:
                cv2.circle(frame, tuple(point), 2, (255, 0, 255), -1)
            #  绘制MAR
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Face with Glasses', frame)
    # cv2.resizeWindow('Face with Glasses', 640, 480)  # 固定主窗口为640x480

    # cv2.imshow('Glasses', bg_roi) # 单独显示眼镜
    # cv2.resizeWindow('Glasses', int(bg_roi.shape[1] * 1.5), int(bg_roi.shape[0] * 1.5))  # 放大1.5倍显示

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        current_sticker_index = (current_sticker_index + 1) % len(sticker_list)
    elif key == ord('p'):
        de_show = not de_show
    elif key == ord('w'):
        show_whiskers = not show_whiskers
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
