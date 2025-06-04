import cv2
import numpy as np
import requests
import random
# from PIL import Image
# import tkinter as tk
# from tkinter import filedialog
import math

# 天气API配置（需要去OpenWeatherMap注册免费API key）
API_KEY = "50412da7c54dec7293740f41cbc2c917"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"

# 特效参数配置
# RAIN_DROPS = 10  # 雨滴数量
SNOWFLAKES = 100  # 雪花数量
CLOUDS = 5        # 云朵数量

RAINY_INTENSITY = 0.4   # 降雨强度
RAINY_WINDFORCE = 0.1
DROPS = 400

MAX_RAINY_INTENSITY = 0.8   # 降雨强度
MAX_RAINY_WINDFORCE = 0.2
MAX_DROPS = 800

# 摄像头参数配置
CAMERA_WIDTH = 600
CAMERA_HEIGHT = 600

"""
class WeatherEffects:
    def __init__(self, camera_width=640, camera_height=480):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        self.h, self.w = camera_height, camera_width

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
"""

"""
下雨特效
"""
class RainyEffects():
    def __init__(self, cap, is_thunder = False):
        print("****rainy_init****")
        self.cap = cap
        self.h, self.w = CAMERA_HEIGHT, CAMERA_WIDTH
        self.is_thunder = is_thunder
        
        # 初始化云朵系统
        self.cloud_img = cv2.imread('src/image/cloud_1.png', cv2.IMREAD_UNCHANGED)
        if self.cloud_img is None:
            raise ValueError("云朵图片加载失败")
        self._preprocess_cloud()

        # 雨滴系统参数
        if is_thunder:
            self.rain_params = {
                'intensity': MAX_RAINY_INTENSITY,
                'wind_force': MAX_RAINY_WINDFORCE,
                'max_drops': MAX_DROPS,
                'drop_speed_range': (8, 15),
                'drop_length_range': (10, 25)
            }
        else:
            self.rain_params = {
                'intensity': RAINY_INTENSITY,
                'wind_force': RAINY_WINDFORCE,
                'max_drops': DROPS,
                'drop_speed_range': (8, 15),
                'drop_length_range': (10, 25)
            }
        self._init_rain_drops()

        # 闪电系统参数
        if self.is_thunder:
            self.lightning_img = cv2.imread('src/image/thunder_2.png', cv2.IMREAD_UNCHANGED)
            self.lightning_img1 = cv2.imread('src/image/thunder_2.png', cv2.IMREAD_UNCHANGED)
            if self.lightning_img is None:
                print("图片加载失败")
            if self.lightning_img1 is None:
                print("图片1加载失败")
            self._init_lightning_system()

    def _init_rain_drops(self):
        """初始化雨滴粒子系统"""
        self.rain_drops = []
        num = int(self.rain_params['max_drops'] * self.rain_params['intensity'])
        for _ in range(num):
            self.rain_drops.append({
                'x': random.uniform(-self.w*0.6, self.w),
                'y': random.uniform(-self.h, 0),
                'speed': random.uniform(*self.rain_params['drop_speed_range']),
                'length': random.randint(*self.rain_params['drop_length_range']),
                'alpha': random.randint(180, 220)
            })

    def _init_lightning_system(self):
        """初始化闪电特效系统"""
        self.lightning_active = False
        self.animation_frame = 0
        self.max_animation_frames = 15
        self.flash_intensity = 0.0
        self.current_pos = (0, 0)
        self.current_scale = 1.0
        
        # 预生成不同形态闪电
        self.preprocessed_lightnings = [
            self._preprocess_lightning(self.lightning_img, angle=0, scale=1.0),
            self._preprocess_lightning(self.lightning_img, angle=15, scale=0.9),
            self._preprocess_lightning(self.lightning_img, angle=-15, scale=1.1),

            self._preprocess_lightning(self.lightning_img1, angle=0, scale=1.0),
            self._preprocess_lightning(self.lightning_img1, angle=15, scale=0.9),
            self._preprocess_lightning(self.lightning_img1, angle=-15, scale=1.1),
        ]

    def _preprocess_lightning(self, img, angle=0, scale=1.0):
        """预处理闪电素材"""
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
        rotated = cv2.warpAffine(
            img, M, (cols, rows),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0,0,0,0)
        )
        return rotated

    def _update_rain_particles(self):
        """更新雨滴物理状态"""
        wind = self.rain_params['wind_force'] * 15
        for drop in self.rain_drops:
            drop['x'] += wind * (drop['speed']/10)
            drop['y'] += drop['speed']
            if drop['y'] < -self.h or drop['x'] < -50 or drop['x'] > self.w + 50:
                drop['x'] = random.uniform(-self.w*0.4, self.w*1.2)
                drop['y'] = random.uniform(-self.h, 0)
                drop['speed'] = random.uniform(*self.rain_params['drop_speed_range'])

    def _preprocess_cloud(self):
        """预处理云朵素材"""
        target_width = self.w // 6
        target_height = self.h // 4
        self.cloud_img = cv2.resize(self.cloud_img, 
                                (target_width, target_height),
                                interpolation=cv2.INTER_LINEAR)
        self.cloud_alpha = self.cloud_img[:, :, 3] / 255.0
        self.cloud_rgb = cv2.cvtColor(self.cloud_img, cv2.COLOR_BGRA2BGR)

    def _blend_cloud(self, frame, i):
        """混合云层到画面"""
        cloud_h, cloud_w = self.cloud_img.shape[:2]
        y_start, x_start = 0, 0 + i * cloud_w
        roi = frame[y_start:y_start+cloud_h, x_start:x_start+cloud_w]
        
        roi_float = roi.astype(float)
        cloud_float = self.cloud_rgb.astype(float)
        blended = (roi_float * (1 - self.cloud_alpha[..., np.newaxis]) + 
                 cloud_float * self.cloud_alpha[..., np.newaxis])
        frame[y_start:y_start+cloud_h, x_start:x_start+cloud_w] = blended.astype(np.uint8)

    def _trigger_lightning(self):
        """随机触发闪电"""
        if not self.lightning_active and random.random() < 0.05:
            self.lightning_active = True
            self.animation_frame = 0
            self.current_pos = (
                random.randint(100, self.w-100),
                80
            )
            self.current_scale = 0.2 + random.random()*0.1
            self.current_lightning = random.choice(self.preprocessed_lightnings)

    def _apply_lightning_flash(self, frame):
        """应用全屏闪光效果"""
        flash_overlay = frame.copy()
        cv2.rectangle(flash_overlay, (0,0), (self.w, self.h), (255, 255, 255), -1)
        alpha = 0.3 * math.sin(math.pi * self.flash_intensity)
        cv2.addWeighted(flash_overlay, alpha, frame, 1 - alpha, 0, frame)

    def _blend_lightning(self, frame):
        """混合闪电到画面"""
        lightning = self.current_lightning
        if self.current_scale != 1.0:
            h, w = lightning.shape[:2]
            lightning = cv2.resize(lightning, (int(self.w*self.current_scale), 
                                             int(self.h*self.current_scale)))

        """ 
        target_width = self.w // 6
        target_height = self.h // 6
        self.cloud_img = cv2.resize(self.cloud_img, 
                                (target_width, target_height),
                                interpolation=cv2.INTER_LINEAR)
        """

        y_start = self.current_pos[1]
        y_end = y_start + lightning.shape[0]
        x_start = self.current_pos[0]
        x_end = x_start + lightning.shape[1]
        
        if y_end > frame.shape[0] or x_end > frame.shape[1]:
            return
            
        roi = frame[y_start:y_end, x_start:x_end]
        l_bgra = cv2.split(lightning)
        alpha = l_bgra[3] / 255.0
        inv_alpha = 1.0 - alpha
        
        bgr_lightning = cv2.merge(l_bgra[:3])
        bgr_lightning = cv2.addWeighted(bgr_lightning, 2.5, 0, 0, 0)
        
        for c in range(3):
            roi[:, :, c] = (alpha * bgr_lightning[:, :, c] + 
                          inv_alpha * roi[:, :, c]).astype(np.uint8)

    def add_effects(self):
        """主效果合成方法"""
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        overlay = np.zeros_like(frame)

        # 更新各子系统
        self._update_rain_particles()
        if hasattr(self, 'lightning_img'):
            self._trigger_lightning()

        # 渲染云层
        for i in range(6):
            self._blend_cloud(frame, i)

        # 渲染雨滴
        for drop in self.rain_drops:
            x, y = int(drop['x']), int(drop['y'])
            end_x = x + int(self.rain_params['wind_force'] * drop['length'])
            end_y = y + drop['length']
            cv2.line(overlay, (x, y), (end_x, end_y), 
                    (200, 200, 255, drop['alpha']), 1, cv2.LINE_AA)
            if random.random() < 0.3:
                cv2.circle(overlay, (end_x, end_y), 1, (255, 255, 255), -1)

        # 渲染闪电
        if self.is_thunder:
            if hasattr(self, 'lightning_img') and self.lightning_active:
                progress = self.animation_frame / self.max_animation_frames
                self.flash_intensity = math.sin(math.pi * progress)
                self._blend_lightning(frame)
                self._apply_lightning_flash(frame)
                self.animation_frame += 1
                if self.animation_frame >= self.max_animation_frames:
                    self.lightning_active = False

        # 最终合成
        frame = cv2.addWeighted(frame, 0.85, overlay, 0.4, 0)
        return cv2.GaussianBlur(frame, (3, 3), 0)

    
"""
下雪特效
"""
class SnowyEffects:
    def __init__(self, cap):
        self.cap = cap
        # self.cap = cv2.VideoCapture(0)
        print("****snowy_init****")
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        self.h, self.w = CAMERA_HEIGHT, CAMERA_WIDTH

    def add_effects(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        """添加雪景特效（使用真实雪花图片）"""
        # 加载雪花图片（需准备带透明通道的PNG）
        snow_img = cv2.imread('src\image\snow_4.png', cv2.IMREAD_UNCHANGED)
        if snow_img is None:
            raise FileNotFoundError("找不到雪花图片")

        # 初始化雪花列表（包含位置和属性）
        if not hasattr(self, 'snowflakes'):
            self.snowflakes = []
            for _ in range(SNOWFLAKES):
                self.snowflakes.append({
                    'pos': [random.randint(-self.w, self.w*2), random.randint(-self.h, 0)],
                    'speed': random.uniform(5, 25),
                    'scale': random.uniform(0.01, 0.03),
                    'drift': random.uniform(5, 10)
                })

        # 更新并绘制每个雪花
        for flake in self.snowflakes:
            # 调整雪花尺寸
            scaled_snow = cv2.resize(snow_img, None, 
                                fx=flake['scale'], 
                                fy=flake['scale'])
            
            # 叠加雪花到当前帧
            frame = self.overlay_transparent(frame, scaled_snow, 
                                        int(flake['pos'][0]), 
                                        int(flake['pos'][1]))
            
            # 更新位置（包含横向飘动）
            flake['pos'][1] += flake['speed']  # 纵向下落速度
            flake['pos'][0] += flake['drift']  # 横向飘动
            
            # 重置位置（当雪花超出屏幕时）
            if flake['pos'][1] > self.h or flake['pos'][0] > self.w:
                flake['pos'] = [
                    random.randint(-self.w, self.w*2),
                    random.randint(-self.h, 0)
                ]
                # flake['speed'] = random.uniform(5, 25)
                # flake['scale'] = random.uniform(0.01, 0.03)
                # flake['drift'] = random.uniform(10, 15)

        return frame
    
    def overlay_transparent(self, bg, overlay_img, x, y):
        if x >= bg.shape[1] or y >= bg.shape[0] or x + overlay_img.shape[1] <= 0 or y + overlay_img.shape[0] <= 0:
            return bg

        # 提取 alpha 通道并分离颜色通道
        alpha = overlay_img[:, :, 3] / 255.0  # 归一化到 [0,1]
        overlay_rgb = overlay_img[:, :, :3]   # 提取 BGR 通道
        
        # 计算有效叠加区域
        y1 = max(y, 0)
        y2 = min(y + overlay_img.shape[0], bg.shape[0])
        x1 = max(x, 0)
        x2 = min(x + overlay_img.shape[1], bg.shape[1])
        
        # 调整 overlay 子区域
        overlay_sub = overlay_rgb[y1 - y:y2 - y, x1 - x:x2 - x]
        alpha_sub = alpha[y1 - y:y2 - y, x1 - x:x2 - x]
        
        # 混合计算
        bg_sub = bg[y1:y2, x1:x2]
        bg[y1:y2, x1:x2] = (1 - alpha_sub[..., None]) * bg_sub + alpha_sub[..., None] * overlay_sub
        return bg


"""
晴天特效
"""
class SunnyEffects:
    def __init__(self, cap):
        self.cap = cap
        # self.cap = cv2.VideoCapture(0)
        print("****sunny_init****")
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        self.h, self.w = CAMERA_HEIGHT, CAMERA_WIDTH
        
        # 初始化云朵
        cloud_path = 'src/image/sunny_2.png'
        self.cloud_img = cv2.imread(cloud_path, cv2.IMREAD_UNCHANGED)
        if self.cloud_img is None: 
            raise ValueError("云朵图片加载失败")
        self.cloud_alpha = None
        self._preprocess_cloud()
        
        # 特效参数
        self.glow_phase = 0.0
        self.sun_pos = (int(0.2*self.w), int(0.15*self.h))
        self.ray_particles = self._init_ray_particles()

    def _init_ray_particles(self):
        """初始化阳光射线:ml-citation{ref="4,5" data="citationList"}"""
        particles = []
        for _ in range(20):
            particles.append({
                'angle': random.uniform(-math.pi/6, math.pi/6),
                'length': random.randint(60, 120),
                'phase': random.uniform(0, 2*math.pi)
            })
        return particles

    def _draw_dynamic_sun(self, frame):
        """绘制动态太阳:ml-citation{ref="4,5" data="citationList"}"""
        # 基础太阳
        cv2.circle(frame, self.sun_pos, 35, 
                  (255, 255, 0), -1)
        
        # 波动光晕
        for i in range(3):
            radius = 50 + 15*i + 10*math.sin(self.glow_phase + i)
            alpha = 0.4 - 0.1*i
            overlay = frame.copy()
            cv2.circle(overlay, self.sun_pos, int(radius),
                      (250, 250, 25), -1)
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        
        # 更新相位
        self.glow_phase += 0.07

    def _draw_sun_rays(self, frame):
        """绘制动态光线:ml-citation{ref="5" data="citationList"}""" 
        overlay = np.zeros_like(frame)
        for p in self.ray_particles:
            length = p['length'] * (0.7 + 0.3*math.sin(p['phase']))
            end = (
                int(self.sun_pos[0] + length*math.cos(p['angle'])),
                int(self.sun_pos[1] + length*math.sin(p['angle']))
            )
            cv2.line(overlay, self.sun_pos, end,
                    (50, 200, 255), 2)
            p['phase'] += 0.05
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    def _preprocess_cloud(self):
        """预处理云朵尺寸和透明度（调整为顶部1/5高度并水平铺满）"""
        # 计算目标尺寸：宽度铺满窗口，高度为窗口1/5
        target_width = self.w
        target_height = self.h // 4
        
        # 调整云朵尺寸（使用线性插值保持质量）
        self.cloud_img = cv2.resize(self.cloud_img, 
                                (target_width, target_height),
                                interpolation=cv2.INTER_LINEAR)
        
        # 分离透明通道
        self.cloud_alpha = self.cloud_img[:, :, 3] / 255.0
        self.cloud_rgb = cv2.cvtColor(self.cloud_img, cv2.COLOR_BGRA2BGR)

    def _blend_cloud(self, frame):
        """在画面顶部混合云层（占1/5高度且水平铺满）:ml-citation{ref="1,3,8"}"""
        # 获取ROI区域
        y_start, x_start = 0, 0
        cloud_h, cloud_w = self.cloud_img.shape[:2]
        roi = frame[y_start:y_start+cloud_h, x_start:x_start+cloud_w]
        
        # 转换数据类型为浮点型
        roi_float = roi.astype(float)
        cloud_float = self.cloud_rgb.astype(float)
        
        # 计算混合结果（手动应用alpha通道）
        blended = (
            roi_float * (1 - self.cloud_alpha[..., np.newaxis]) + 
            cloud_float * self.cloud_alpha[..., np.newaxis]
        )
        
        # 回写结果并转换数据类型
        frame[y_start:y_start+cloud_h, 
            x_start:x_start+cloud_w] = blended.astype(np.uint8)


    def add_effects(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 应用特效
        self._draw_dynamic_sun(frame)
        self._draw_sun_rays(frame)
        self._preprocess_cloud()
        self._blend_cloud(frame)
        
        # 增强光照效果:ml-citation{ref="4,6" data="citationList"}
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1]*1.2, 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2]*1.1, 0, 255)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return frame

"""
多云特效
"""
class CloudyEffects:
    def __init__(self, cap):
        self.cap = cap
        # self.cap = cv2.VideoCapture(0)
        print("****cloudy_init****")
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        self.h, self.w = CAMERA_HEIGHT, CAMERA_WIDTH

    def add_effects(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        """添加多云特效"""


    
"""获取天气数据"""
def get_weather(city_name):
    
    url = BASE_URL + "q=" + city_name + "&appid=" + API_KEY
    response = requests.get(url).json()
    weather = response['weather'][0]['main'].lower()
    return weather

# 打开文件对话框选择图片文件
"""选择图片文件"""
"""
def select_image():
    
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path
"""

def main():
    # 选择图片文件
    # img_path = select_image()
    """
    img_path = 'src/image/photo.jpg'
    print(img_path)
    if not img_path:
        print("未选择图片")
        return
    """

    # 获取天气信息
    city = "Shenzhen"
    # weather = get_weather(city)
    weather = 'rain'
    print(weather)
    
    # 初始化背景
    # pil_img = Image.open(img_path).convert('RGB')
    # effect = WeatherEffects(pil_img)
    """
    rain_effect = RainyEffects(pil_img)
    snow_effect = SnowyEffects(pil_img)
    sunny_effect = SunnyEffects(pil_img)
    cloudy_effect = CloudyEffects(pil_img)
    
    """

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    rain_effect = RainyEffects(cap)
    thunder_effect = RainyEffects(cap, True)
    snow_effect = SnowyEffects(cap)
    sunny_effect = SunnyEffects(cap)
    cloudy_effect = CloudyEffects(cap)
    
    # 创建窗口
    cv2.namedWindow('Weather Mirror', cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('Weather Mirror', cv2.WND_PROP_TOPMOST, 1)
    
    while True:
        # frame = effect.background.copy()
        
        if weather == 'rain':
            # frame = effect.add_rain(frame)
            # frame = rain_effect.background.copy()
            # rain_effect = RainyEffects(cap)
            frame = rain_effect.add_effects()
        elif weather == 'snow':
            # frame = snow_effect.background.copy()
            # snow_effect = SnowyEffects(cap)
            frame = snow_effect.add_effects()
        elif weather == 'clouds':
            # frame = cloudy_effect.background.copy()
            # sunny_effect = SunnyEffects(cap)
            frame = cloudy_effect.add_effects()
        elif weather == 'thunderrain':
            RAINY_INTENSITY = 0.8
            RAINY_WINDFORCE = 0.4
            MAX_DROPS = 800
            frame = thunder_effect.add_effects()
        else:  # 默认晴天
            # frame = sunny_effect.background.copy()
            # cloudy_effect = CloudyEffects(cap)
            frame = sunny_effect.add_effects()
        
        cv2.imshow('Weather Camera', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.setWindowProperty('Weather Camera', cv2.WND_PROP_TOPMOST, 1)
        
        # 按下ESC退出
        if cv2.waitKey(30) == 27 or cv2.getWindowProperty('Weather Camera', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
