import cv2
import requests
from AddWeather import RainyEffects, SnowyEffects, SunnyEffects, CloudyEffects

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

def main():
    # 获取天气信息
    city = "Shenzhen"
    # weather = get_weather(city)
    weather = 'thunderrain'
    print(weather)

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
            frame = rain_effect.add_effects()
        elif weather == 'snow':
            frame = snow_effect.add_effects()
        elif weather == 'clouds':
            frame = cloudy_effect.add_effects()
        elif weather == 'thunderrain':
            frame = thunder_effect.add_effects()
        else:  # 默认晴天
            frame = sunny_effect.add_effects()
        
        cv2.imshow('Weather Camera', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.setWindowProperty('Weather Camera', cv2.WND_PROP_TOPMOST, 1)
        
        # 按下ESC退出
        if cv2.waitKey(30) == 27 or cv2.getWindowProperty('Weather Camera', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
