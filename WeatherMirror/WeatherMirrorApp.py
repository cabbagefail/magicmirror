import cv2
import requests
from AddWeather import RainyEffects, SnowyEffects, SunnyEffects, CloudyEffects
import mediapipe as mp
import numpy as np
import time

# 天气API配置（需要去OpenWeatherMap注册免费API key）
API_KEY = "50412da7c54dec7293740f41cbc2c917"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"

# 摄像头参数配置
CAMERA_WIDTH = 600
CAMERA_HEIGHT = 600

    
"""获取天气数据"""
def get_weather(city_name):
    
    url = BASE_URL + "q=" + city_name + "&appid=" + API_KEY
    response = requests.get(url).json()
    weather = response['weather'][0]['main'].lower()
    return weather

# 表情分类
def classify_expression(landmarks):
    # 提取关键点：嘴角、眉毛、眼睛
    lm = landmarks

    # 获取嘴巴相关坐标
    mouth_left = np.array([lm[61].x, lm[61].y])
    mouth_right = np.array([lm[291].x, lm[291].y])
    mouth_top = np.array([lm[13].x, lm[13].y])
    mouth_bottom = np.array([lm[14].x, lm[14].y])
    mouth_width = np.linalg.norm(mouth_right - mouth_left)
    mouth_open = np.linalg.norm(mouth_bottom - mouth_top)

    # 获取眉毛高度
    brow_left = np.array([lm[70].x, lm[70].y])
    eye_left_top = np.array([lm[159].x, lm[159].y])
    brow_eye_diff = brow_left[1] - eye_left_top[1]

    # 获取眼睛开合程度（判断哭泣）
    eye_top = np.array([lm[159].x, lm[159].y])
    eye_bottom = np.array([lm[145].x, lm[145].y])
    eye_open = np.linalg.norm(eye_top - eye_bottom)

    # 默认表情参数：mouth_open=0.0006 mouth_width=0.095 eye_open=0.029 brow_eye_diff=-0.041
    # print("mouth_width:" + str(mouth_width))# 生气参数：0.081
    print("mouth_open:" + str(mouth_open) + " " + "eye_open:" + str(eye_open))# 哭泣参数：0.06 0.008
    # print("mouth_open:" + str(mouth_open) + " " + "mouth_width:" + str(mouth_width))# 微笑参数：0.027 0.135
    # print("brow_eye_diff" + str(brow_eye_diff))# 悲伤参数：

    # 判断规则（可调整阈值）
    if mouth_width < 0.09:
        return "生气"
    elif mouth_open > 0.04 and eye_open < 0.012:
        return "哭泣"
    elif mouth_open > 0.02 and mouth_width > 0.13:
        return "微笑"
    elif brow_eye_diff > 0.035:
        return "轻度悲伤"
    else:
        return "默认"
    
def facial_expression(frame, face_mesh):
    results = face_mesh.process(frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            expression = classify_expression(face_landmarks.landmark)
            print(expression)
            
    return expression

# 入口函数
def main():
    # 获取天气信息
    city = "Shenzhen"
    # weather = get_weather(city)
    weather = 'thunderrain'
    print(weather)

    # 开启摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    # 初始化天气特效类
    rain_effect = RainyEffects(cap)
    thunder_effect = RainyEffects(cap, True)
    snow_effect = SnowyEffects(cap)
    sunny_effect = SunnyEffects(cap)
    cloudy_effect = CloudyEffects(cap)
    
    # 创建窗口
    cv2.namedWindow('Weather Mirror', cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('Weather Mirror', cv2.WND_PROP_TOPMOST, 1)

    # Mediapipe 初始化
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    # 控制调用频率的变量
    last_check_time = 0
    expression_check_interval = 3  # 每3秒检测一次表情
    current_expression = '默认'  # 缓存上次检测到的结果
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_raw = frame.copy()

        current_time = time.time()
        if current_time - last_check_time >= expression_check_interval:
            current_expression = facial_expression(frame_raw, face_mesh)
            last_check_time = current_time
        print(current_expression)
        
        """
        if weather == 'rain':
            frame = rain_effect.add_effects(frame)
        elif weather == 'snow':
            frame = snow_effect.add_effects(frame)
        elif weather == 'clouds':
            frame = cloudy_effect.add_effects(frame)
        elif weather == 'thunderrain':
            frame = thunder_effect.add_effects(frame)
        else:  # 默认晴天
            frame = sunny_effect.add_effects(frame)
        """

        if current_expression == '悲伤':
            frame = rain_effect.add_effects(frame)
        elif current_expression == '生气':
            frame = snow_effect.add_effects(frame)
        elif current_expression == '微笑':
            frame = sunny_effect.add_effects(frame)
        elif current_expression == '哭泣':
            frame = thunder_effect.add_effects(frame)
        else:  # 默认天气
            frame = cloudy_effect.add_effects(frame)

        cv2.imshow('Weather Camera', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.setWindowProperty('Weather Camera', cv2.WND_PROP_TOPMOST, 1)
        
        # 按下ESC退出
        if cv2.waitKey(30) == 27 or cv2.getWindowProperty('Weather Camera', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
