import cv2
import numpy as np
import math
import random
import time


# 计算旋转后能包含完整图像的最小矩形尺寸
def get_rotated_bounding_box(image_width, image_height, angle_degrees):
    """计算旋转后能包含完整图像的最小矩形尺寸"""
    angle = np.radians(angle_degrees)
    # 计算四个顶点坐标 (左上角、右上角、右下角、左下角)
    corners = np.array([
        [0, 0],
        [image_width, 0],
        [image_width, image_height],
        [0, image_height]
    ])

    # 旋转矩阵
    center = np.array([image_width / 2, image_height / 2])
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    # 计算旋转后的坐标
    rotated_corners = np.dot(corners - center, rotation_matrix.T) + center

    # 计算包围盒
    min_x, min_y = np.min(rotated_corners, axis=0)
    max_x, max_y = np.max(rotated_corners, axis=0)

    return int(max_x - min_x), int(max_y - min_y), (min_x, min_y)


# 获取左眼中心点
def left_eye_center(landmarks):
    left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    return np.mean(left_eye_points, axis=0).astype(int)


# 获取右眼中心点
def right_eye_center(landmarks):
    right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
    return np.mean(right_eye_points, axis=0).astype(int)


# 计算EAR
def calculate_ear(eye_points):
    # 计算眼睛的垂直距离（上眼皮与下眼皮之间的距离）
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    # 计算眼睛的水平距离（左右眼角之间的距离）
    C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    # 计算 EAR
    ear = (A + B) / (2.0 * C)
    return ear


# 获取嘴巴中心点
# def mouth_center():
#     mouth_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
#     return np.mean(mouth_points, axis=0).astype(int)


# 计算嘴巴张开度（垂直距离/水平距离）
def calculate_mar(mouth_points):
    # 垂直距离：上唇下缘到下唇上缘
    vertical_dist = np.linalg.norm(np.array(mouth_points[13]) - np.array(mouth_points[19]))
    # 水平距离：嘴角左右距离
    horizontal_dist = np.linalg.norm(np.array(mouth_points[0]) - np.array(mouth_points[6]))
    return vertical_dist / horizontal_dist


# 添加贴纸
def apply_sticker(frame, landmarks, current_sticker_index, sticker_list, scale_factors):
    # 获取眼部关键点坐标（使用68点模型中的36-47号点）
    left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

    # 计算眼睛中心坐标
    left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
    right_eye_center = np.mean(right_eye_points, axis=0).astype(int)

    # 计算眼睛间距和旋转角度
    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    eye_distance = np.hypot(dx, dy)  # 眼睛间距
    angle = -np.degrees(np.arctan2(dy, dx))  # 计算角度

    # 根据眼睛间距缩放眼镜
    sticker_img = sticker_list[current_sticker_index]
    scale_factor = eye_distance / (sticker_img.shape[1] * scale_factors[current_sticker_index])  # 调整系数使眼镜匹配眼睛
    safe_scale = scale_factor * 0.8  # 留出20%安全边距
    resized_glass = cv2.resize(sticker_img, None, fx=scale_factor, fy=scale_factor,
                               interpolation=cv2.INTER_AREA)

    # 旋转眼镜
    (h, w) = resized_glass.shape[:2]

    # 计算旋转参数
    new_w, new_h, offset = get_rotated_bounding_box(w, h, angle)
    center = (w // 2, h // 2)

    # 创建旋转矩阵并进行平移补偿
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_w - w) / 2  # x轴平移补偿
    M[1, 2] += (new_h - h) / 2  # y轴平移补偿

    # 执行旋转
    rotated_glass = cv2.warpAffine(
        resized_glass, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    # 更新尺寸参数
    (h, w) = rotated_glass.shape[:2]

    # 计算眼镜位置
    glasses_center = (
        int((left_eye_center[0] + right_eye_center[0]) / 2),
        int((left_eye_center[1] + right_eye_center[1]) / 2 + eye_distance * 0.05)  # 下移15%眼距
    )

    # 计算覆盖区域
    x_offset = glasses_center[0] - rotated_glass.shape[1] // 2
    y_offset = glasses_center[1] - rotated_glass.shape[0] // 2

    # 使用alpha通道进行图像融合
    if rotated_glass.shape[2] == 4:
        alpha = rotated_glass[:, :, 3] / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])
        glass_bgr = rotated_glass[:, :, :3]
    else:
        alpha = np.ones_like(rotated_glass[:, :, 0], dtype=np.float32)
        glass_bgr = rotated_glass

    # 处理边界情况
    y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + h)
    x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + w)

    if y2 - y1 <= 0 or x2 - x1 <= 0:
        return

    # 裁剪ROI区域
    glass_roi = glass_bgr[0:y2 - y1, 0:x2 - x1]
    alpha_roi = alpha[0:y2 - y1, 0:x2 - x1]
    bg_roi = frame[y1:y2, x1:x2]  # 背景ROI

    # 融合图像
    bg_roi[:] = (glass_roi * alpha_roi + bg_roi * (1 - alpha_roi)).astype(np.uint8)


# 添加猫咪胡须
def apply_whiskers(frame, landmarks, whiskers_img, scale_factor=1.5):
    """
    在人脸鼻子和嘴巴之间添加猫咪胡须贴图，并根据人脸角度进行旋转。

    参数:
        frame (np.ndarray): 当前视频帧图像
        landmarks (dlib.full_object_detection): 面部关键点检测结果
        whiskers_img (np.ndarray): 猫咪胡须图像（含 alpha 通道）
        scale_factor (float): 贴图缩放比例（建议 0.6~1.2）

    返回:
        np.ndarray: 添加了猫咪胡须的图像
    """
    if whiskers_img is None or whiskers_img.shape[2] != 4:
        return frame

    # 获取眼部关键点用于计算角度
    left_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], axis=0).astype(int)
    right_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], axis=0).astype(int)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = -np.degrees(np.arctan2(dy, dx))  # 计算旋转角度

    # 获取鼻尖和上唇中点用于定位贴图
    nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
    top_lip = (landmarks.part(51).x, landmarks.part(51).y)

    # 垂直居中于鼻子与上唇之间
    whisker_y = int((nose_tip[1] + top_lip[1]) / 2)

    # Z轴偏移量，让胡须更贴近鼻孔下方
    eye_distance = np.hypot(dx, dy)
    depth_offset = int(eye_distance * 0.03)
    whisker_y -= depth_offset  # 上提一点，模拟“内陷”效果

    # 缩放贴图（根据眼距自动适配）
    h, w = whiskers_img.shape[:2]
    desired_width = int(eye_distance * scale_factor)  # 胡须宽度 ≈ 眼睛间距
    resized_whiskers = cv2.resize(whiskers_img, None, fx=desired_width / w, fy=desired_width / w,
                                  interpolation=cv2.INTER_AREA)

    # 旋转贴图
    (h_r, w_r) = resized_whiskers.shape[:2]
    center = (w_r // 2, h_r // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算旋转后包围盒大小以避免裁剪
    new_w, new_h, offset = get_rotated_bounding_box(w_r, h_r, angle)
    M[0, 2] += (new_w - w_r) / 2  # x轴平移补偿
    M[1, 2] += (new_h - h_r) / 2  # y轴平移补偿

    rotated_whiskers = cv2.warpAffine(
        resized_whiskers,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    # 更新尺寸参数
    (h_r, w_r) = rotated_whiskers.shape[:2]

    # 定位贴图中心位置（水平居中于鼻尖正下方）
    whisker_x = int(nose_tip[0] - w_r // 2)
    whisker_position = (whisker_x, whisker_y - h_r // 2)

    # ROI区域限制
    x1, y1 = whisker_position
    x2, y2 = x1 + w_r, y1 + h_r

    # 边界检查
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        return frame

    roi = frame[y1:y2, x1:x2]

    # 使用alpha通道融合图像
    if rotated_whiskers.shape[2] == 4:
        alpha_channel = rotated_whiskers[:, :, 3] / 255.0  # 提取并归一化 alpha 通道
        alpha_mask = cv2.merge([alpha_channel, alpha_channel, alpha_channel])  # 扩展为三通道
        whisker_bgr = rotated_whiskers[:, :, :3]  # 分离 BGR 通道
    else:
        alpha_mask = np.ones_like(rotated_whiskers[:, :, 0], dtype=np.float32)  # 如果没有 alpha 通道，创建全 1 的 mask
        whisker_bgr = rotated_whiskers

    # 融合图像
    roi[:] = (whisker_bgr * alpha_mask + roi * (1 - alpha_mask)).astype(np.uint8)

    return frame


'''
表情识别
'''


# 新增函数：计算眉毛倾斜角度
def calculate_eyebrow_angle(landmarks, side='both'):
    """
    计算眉毛倾斜角度（单位：度）
    side: 'left', 'right', 或 'both'
    """

    def calc_single_angle(points):
        # 使用绝对值计算dx和dy
        dx = abs(points[1][0] - points[0][0])
        dy = abs(points[1][1] - points[0][1])
        return math.degrees(math.atan2(dy, dx))

    # 左眉使用点17和20
    left_points = [
        (landmarks.part(17).x, landmarks.part(17).y),
        (landmarks.part(20).x, landmarks.part(20).y)
    ]

    # 右眉使用点23和26
    right_points = [
        (landmarks.part(23).x, landmarks.part(23).y),
        (landmarks.part(26).x, landmarks.part(26).y)
    ]

    if side == 'left':
        return calc_single_angle(left_points)
    elif side == 'right':
        return calc_single_angle(right_points)
    else:
        return (calc_single_angle(left_points) + calc_single_angle(right_points)) / 2


# 新增函数：计算嘴角角度
def calculate_lip_corner_angle(landmarks):
    """
    计算嘴角角度（单位：度）
    """
    # 左嘴角 (48号点) 和 右嘴角 (54号点)
    left_corner = (landmarks.part(48).x, landmarks.part(48).y)
    right_corner = (landmarks.part(54).x, landmarks.part(54).y)

    # 上唇中点 (51号点)
    top_lip = (landmarks.part(51).x, landmarks.part(51).y)

    # 下唇中点 (57号点)
    bottom_lip = (landmarks.part(57).x, landmarks.part(57).y)

    # 计算左嘴角角度
    dx_left = bottom_lip[0] - left_corner[0]
    dy_left = bottom_lip[1] - left_corner[1]
    angle_left = math.degrees(math.atan2(dy_left, dx_left))

    # 计算右嘴角角度
    dx_right = right_corner[0] - bottom_lip[0]
    dy_right = -(right_corner[1] - bottom_lip[1])
    angle_right = math.degrees(math.atan2(dy_right, dx_right))

    return (angle_left + angle_right) / 2


# 新增函数：表情分类
def classify_emotion(mar, ear, eyebrow_angle, lip_angle, ear_derivative=0, mar_derivative=0):
    """
    基于特征值分类表情（增强条件区分度）
    参数:
        mar: 嘴巴纵横比
        ear: 眼睛纵横比
        eyebrow_angle: 眉毛角度（正值下垂，负值上扬）
        lip_angle: 嘴角角度
        ear_derivative: 眼睛纵横比变化率（可选，不完善）
        mar_derivative: 嘴巴纵横比变化率（可选，不完善）
    返回: 'neutral', 'happy', 'sad', 'surprise', 'angry'
    """

    # 惊讶检测（纵向张嘴 + 睁眼）
    if (0.4 < mar and 0.27 < ear and 40 < lip_angle < 60) or (0.2 < mar_derivative and 0.15 < ear_derivative):
        return 'surprise'

    # 快乐检测（中等张嘴 + 嘴角上扬）
    elif 0.15 < mar < 0.4 and 0.21 < ear < 0.39 and 36 < lip_angle < 50 and eyebrow_angle > 10:
        return 'happy'

    # 生气检测（嘴角下垂 + 眼睛瞪大）
    elif 0.01 < mar < 0.1 and 0.29 < ear and 20 < lip_angle < 40:
        return 'angry'

    # 悲伤检测（闭嘴 + 嘴角下垂 + 眼睛微闭）
    elif mar < 0.15 and ear < 0.23 and lip_angle < 40 or eyebrow_angle < 11:
        return 'sad'

    # 中性表情（默认）
    else:
        return 'neutral'


'''
天气特效
'''


class WeatherEffectSystem:
    def __init__(self, width, height):
        """初始化天气特效系统"""
        self.width = width
        self.height = height
        self.current_weather = 'normal'
        self.weather_params = {
            'sunny': {  # 晴天特效
                'filter': (30, 100, 255, 0.1),  # 暖色调
                'particles': 'sun',
                'particle_count': 50,
                'speed': 1,
                'special': 'sunshine'
            },
            'light_rain': {  # 小雨特效
                'filter': (120, 180, 220, 0.15),
                'particles': 'rain',
                'particle_count': 100,
                'speed': 5,
                'special': None
            },
            'snow': {  # 下雪特效
                'filter': (200, 220, 255, 0.18),
                'particles': 'snow',
                'particle_count': 200,
                'speed': 2,
                'special': 'wind'
            },
            'thunderstorm': {  # 大雨+打雷
                'filter': (80, 130, 200, 0.25),
                'particles': 'rain',
                'particle_count': 300,
                'speed': 10,
                'special': 'lightning'
            },
            'normal': {  # 默认天气
                'filter': (0, 0, 0, 0.0),
                'particles': None,
                'particle_count': 0,
                'speed': 0,
                'special': None
            }
        }
        self.particles = []
        self.last_lightning = 0
        self.lightning_duration = 0

    def set_weather(self, weather):
        valid_weathers = ['sunny', 'light_rain', 'thunderstorm', 'snow', 'normal']
        if weather not in valid_weathers:
            weather = 'normal'
        if weather != self.current_weather:
            self.current_weather = weather
            self.particles = []
            self._init_particles()

    def _init_particles(self):
        """初始化粒子"""
        params = self.weather_params[self.current_weather]
        if params['particles'] == 'rain':
            self.particles = [
                {'x': random.randint(0, self.width),
                 'y': random.randint(-50, 0),
                 'length': random.randint(5, 15),
                 'speed': params['speed']}
                for _ in range(params['particle_count'])
            ]
        elif params['particles'] == 'snow':
            self.particles = [
                {'x': random.randint(0, self.width),
                 'y': random.randint(-50, 0),
                 'size': random.randint(2, 6),
                 'speed': params['speed']}
                for _ in range(params['particle_count'])
            ]

    def update(self):
        """更新天气特效状态"""
        current_time = time.time()

        # 更新粒子位置
        if self.weather_params[self.current_weather]['particles']:
            for p in self.particles:
                p['y'] += p['speed']
                if p['y'] > self.height:
                    p['y'] = random.randint(-50, 0)
                    p['x'] = random.randint(0, self.width)

            # 特殊效果处理
            if self.weather_params[self.current_weather].get('special') == 'lightning':
                # 随机触发闪电（每秒约1次）
                if current_time - self.last_lightning > random.uniform(1, 3):
                    self.last_lightning = current_time
                    self.lightning_duration = 0.2
                    # 可以在这里添加声音播放逻辑

            # 风效处理
            if self.weather_params[self.current_weather].get('special') == 'wind':
                wind_strength = random.uniform(-0.5, 0.5)
                for p in self.particles:
                    p['x'] += wind_strength

        # 减少闪电持续时间
        if self.lightning_duration > 0:
            self.lightning_duration -= 0.016  # 假设每帧16ms

    def apply_effects(self, frame):
        """应用所有天气特效到帧"""
        # 应用天气滤镜
        frame = self._apply_weather_filter(frame)

        # 应用粒子特效
        if self.weather_params[self.current_weather]['particles']:
            if self.weather_params[self.current_weather]['particles'] == 'rain':
                frame = self._draw_rain(frame)
            elif self.weather_params[self.current_weather]['particles'] == 'snow':
                frame = self._draw_snow(frame)

        # 应用特殊效果
        if self.weather_params[self.current_weather].get('special') == 'lightning' and self.lightning_duration > 0:
            frame = self._draw_lightning(frame)

        return frame

    def _apply_weather_filter(self, frame):
        """应用天气色彩滤镜"""
        b, g, r, alpha = self.weather_params[self.current_weather]['filter']
        if alpha <= 0:
            return frame

        overlay = np.zeros_like(frame)
        overlay[:] = [b, g, r]

        # 使用HSV混合模式
        hsv_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2HSV).astype("float32")
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype("float32")

        hsv_frame[..., 0] = hsv_overlay[..., 0]  # 替换色调
        hsv_frame[..., 1] = np.clip(hsv_frame[..., 1] * (1 - alpha / 2), 0, 255)  # 降低饱和度
        hsv_frame[..., 2] = np.clip(hsv_frame[..., 2] * (1 - alpha / 3), 0, 255)  # 降低亮度

        result = cv2.cvtColor(hsv_frame.astype("uint8"), cv2.COLOR_HSV2BGR)

        # 添加噪点增强真实感
        if self.current_weather in ['surprise']:
            noise = np.random.normal(0, 2, frame.shape).astype("uint8")
            result = cv2.add(result, noise)

        return result

    def _draw_rain(self, frame):
        """绘制雨滴效果"""
        for p in self.particles:
            end_y = int(p['y'] + p['length'])
            end_y = min(end_y, self.height)
            cv2.line(frame, (int(p['x']), int(p['y'])),
                     (int(p['x']), end_y),
                     (180, 180, 255), 1)
        return frame

    def _draw_snow(self, frame):
        """绘制雪花效果"""
        for p in self.particles:
            cv2.circle(frame, (int(p['x']), int(p['y'])),
                       p['size'], (255, 255, 255), -1)
        return frame

    def _draw_lightning(self, frame):
        """绘制闪电效果"""
        # 主闪电
        cv2.line(frame, (self.width // 3, 0), (self.width // 2, self.height),
                 (255, 255, 200), 3)
        # 分支闪电
        if random.random() < 0.3:
            branch_point = (self.width // 2 - 20, self.height // 2)
            branch_end = (self.width // 2 + 30, self.height // 2 + 50)
            cv2.line(frame, branch_point, branch_end, (255, 255, 200), 2)
        return frame
