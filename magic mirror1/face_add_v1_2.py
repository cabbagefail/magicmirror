import cv2
import dlib
import numpy as np


# # 初始化dlib的人脸检测器和关键点预测器
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")  # 需下载模型文件

# # 加载眼镜图像（带alpha通道）
# glass_img = cv2.imread("images/glass.png", cv2.IMREAD_UNCHANGED)
#
# # 设置视频捕获
# cap = cv2.VideoCapture(0)


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

    # 新增：Z轴偏移量，让胡须更贴近鼻孔下方
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




# 眼镜图像四个角点（按左上、右上、右下、左下顺序）
# glass_points = np.array([
#     [0, 0],                  # 左上角
#     [glass_img.shape[1], 0], # 右上角
#     [glass_img.shape[1], glass_img.shape[0]], # 右下角
#     [0, glass_img.shape[0]]  # 左下角
# ], dtype=np.float32)


# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # 转换为灰度图像用于人脸检测
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # 人脸检测
#     faces = detector(gray, 0)
#
#     for face in faces:
#         # 关键点检测
#         landmarks = predictor(gray, face)
#
#         # 获取眼部关键点坐标（使用68点模型中的36-47号点）
#         left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
#         right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
#
#         # 计算眼睛中心坐标
#         left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
#         right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
#
#         # 计算眼睛间距和旋转角度
#         dx = right_eye_center[0] - left_eye_center[0]
#         dy = right_eye_center[1] - left_eye_center[1]
#         eye_distance = np.hypot(dx, dy)  # 眼睛间距
#         angle = -np.degrees(np.arctan2(dy, dx))  # 计算角度
#
#         # 根据眼睛间距缩放眼镜
#         scale_factor = eye_distance / (glass_img.shape[1] * 0.45)  # 调整系数使眼镜匹配眼睛
#         safe_scale = scale_factor * 0.8  # 留出20%安全边距
#         resized_glass = cv2.resize(glass_img, None, fx=scale_factor, fy=scale_factor,
#                                    interpolation=cv2.INTER_AREA)
#
#         # 旋转眼镜
#         (h, w) = resized_glass.shape[:2]
#
#         # 计算旋转参数
#         new_w, new_h, offset = get_rotated_bounding_box(w, h, angle)
#         center = (w // 2, h // 2)
#
#         # 创建旋转矩阵并进行平移补偿
#         M = cv2.getRotationMatrix2D(center, angle, 1.0)
#         M[0, 2] += (new_w - w) / 2  # x轴平移补偿
#         M[1, 2] += (new_h - h) / 2  # y轴平移补偿
#
#         # 执行旋转
#         rotated_glass = cv2.warpAffine(
#             resized_glass, M, (new_w, new_h),
#             flags=cv2.INTER_LINEAR,
#             borderMode=cv2.BORDER_CONSTANT,
#             borderValue=(0, 0, 0, 0)
#         )
#
#         # 更新尺寸参数
#         (h, w) = rotated_glass.shape[:2]
#
#         # 计算眼镜位置
#         glasses_center = (
#             int((left_eye_center[0] + right_eye_center[0]) / 2),
#             int((left_eye_center[1] + right_eye_center[1]) / 2 + eye_distance * 0.05)  # 下移15%眼距
#         )
#
#         # 计算覆盖区域
#         x_offset = glasses_center[0] - rotated_glass.shape[1] // 2
#         y_offset = glasses_center[1] - rotated_glass.shape[0] // 2
#
#         # 使用alpha通道进行图像融合
#         if rotated_glass.shape[2] == 4:
#             alpha = rotated_glass[:, :, 3] / 255.0
#             alpha = cv2.merge([alpha, alpha, alpha])
#             glass_bgr = rotated_glass[:, :, :3]
#         else:
#             alpha = np.ones_like(rotated_glass[:, :, 0], dtype=np.float32)
#             glass_bgr = rotated_glass
#
#         # 处理边界情况
#         y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + h)
#         x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + w)
#
#         if y2 - y1 <= 0 or x2 - x1 <= 0:
#             continue
#
#         # 裁剪ROI区域
#         glass_roi = glass_bgr[0:y2 - y1, 0:x2 - x1]
#         alpha_roi = alpha[0:y2 - y1, 0:x2 - x1]
#         bg_roi = frame[y1:y2, x1:x2]  # 背景ROI
#
#         # 融合图像
#         bg_roi[:] = (glass_roi * alpha_roi + bg_roi * (1 - alpha_roi)).astype(np.uint8)
#
#         # 绘制眼部关键点（共12个点）
#         for point in left_eye_points + right_eye_points:
#             cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)  # 绿色小圆点
#
#         # 绘制眼睛中心点
#         cv2.circle(frame, tuple(left_eye_center), 3, (255, 0, 0), -1)  # 蓝色圆点
#         cv2.circle(frame, tuple(right_eye_center), 3, (255, 0, 0), -1)
#
#         # 绘制眼镜中心线（连接两眼中心）
#         cv2.line(frame, tuple(left_eye_center), tuple(right_eye_center), (0, 0, 255), 1)
#
#     cv2.imshow('Face with Glasses', frame)
#     cv2.resizeWindow('Face with Glasses', 640, 480)  # 固定主窗口为640x480
#
#     cv2.imshow('Glasses', bg_roi)
#     cv2.resizeWindow('Glasses', int(bg_roi.shape[1] * 1.5), int(bg_roi.shape[0] * 1.5))  # 放大1.5倍显示
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
