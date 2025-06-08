from PyQt5.QtWidgets import QMainWindow
from cameraui import Ui_widget  # 你的UI模块
import sys
import cv2
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import dlib
from face_add_v1_2 import calculate_ear, apply_sticker, apply_whiskers
from facemain import calculate_mar
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
import json
from WeatherMirrorApp import SunnyEffects,SnowyEffects,RainyEffects,get_weather
# 初始化dlib的人脸检测器和关键点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")  # 需下载模型文件


class Camera(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_widget()  # 加载UI
        self.ui.setupUi(self)
        self.ui.comboBox.currentIndexChanged.connect(self.change_sticker)
        self.cap = cv2.VideoCapture(0)  # 打开摄像头
        self.weather=get_weather("Shenzhen")
        print(self.weather)
        if not self.cap.isOpened():
            print("无法打开摄像头")
            sys.exit(-1)
        self.effect=SunnyEffects(self.cap)
        if self.weather == 'rain':
            self.effect = RainyEffects(self.cap)
        elif self.weather == 'thunderrain':
            self.effect = RainyEffects(self.cap, True)
        elif self.weather == 'snow':
            self.effect = SnowyEffects(self.cap)
        # 初始化贴纸和检测参数
        self.init_stickers()
        self.init_detection_params()
        self.network_manager = QNetworkAccessManager(self)
        # 设置定时器，每30ms更新一帧
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.time_timer=QTimer(self)
        self.time_timer.timeout.connect(self.update_current_time)
        self.time_timer.start(1000)

        self.weather_timer = QTimer(self)  # 天气更新(30分钟)
        self.weather_timer.timeout.connect(self.display_weather)
        self.weather_timer.start(1800000)

        self.display_weather()

    def change_sticker(self, index):
        """根据comboBox的选择切换贴纸"""
        self.current_sticker_index = index
        print(f"已切换至贴纸: {self.ui.comboBox.currentText()}")

    def init_stickers(self):
        """初始化贴纸资源"""
        self.glass_img = cv2.imread("images/glass.png", cv2.IMREAD_UNCHANGED)
        self.glass_2_img = cv2.imread("images/glass_2.png", cv2.IMREAD_UNCHANGED)
        self.glass_3_img = cv2.imread("images/glass_3.png", cv2.IMREAD_UNCHANGED)
        self.whiskers_img = cv2.imread("images/cat_1.png", cv2.IMREAD_UNCHANGED)
        self.sticker_list = [self.glass_img, self.glass_2_img, self.glass_3_img]
        self.scale_factors = [0.45, 0.48, 0.33]
        self.current_sticker_index = 0
        self.show_whiskers = False

    def init_detection_params(self):
        """初始化检测参数"""
        self.EAR_THRESHOLD = 0.2
        self.EAR_CONSEC_FRAMES = 4
        self.blink_counter = 0
        self.blink_history = []
        self.MAR_THRESHOLD = 0.25
        self.MAR_CONSEC_FRAMES = 5
        self.smile_counter = 0
        self.timeout_threshold = 3
        self.last_smile_time = 0

    def update_frame(self):
        """更新摄像头帧并显示在UI上"""
        ret, frame = self.cap.read()
        if not ret:
            return # 统一转为RGB
        frame = self.effect.add_effects()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = detector(gray, 0)

        for face in faces:
            landmarks = predictor(gray, face)

            # 应用贴纸
            apply_sticker(frame, landmarks, self.current_sticker_index,
                          self.sticker_list, self.scale_factors)

            # 微笑检测
            mouth_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
            mar = calculate_mar(mouth_points)

            if mar > self.MAR_THRESHOLD:
                self.smile_counter += 1
                if self.smile_counter >= self.MAR_CONSEC_FRAMES:
                    self.show_whiskers = True
                    self.last_smile_time = cv2.getTickCount()
                    print("检测到微笑！已激活猫咪胡须")
                    self.smile_counter = 0
            else:
                self.smile_counter = 0
                if self.show_whiskers:
                    time_since_last_smile = (cv2.getTickCount() - self.last_smile_time) / cv2.getTickFrequency()
                    if time_since_last_smile > self.timeout_threshold:
                        self.show_whiskers = False
                        print("超时未微笑，已关闭猫咪胡须")

            # 应用胡须特效
            if self.show_whiskers:
                apply_whiskers(frame, landmarks, self.whiskers_img, scale_factor=2.3)

            # 眨眼检测
            left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < self.EAR_THRESHOLD:
                self.blink_counter += 1
            else:
                if self.blink_counter >= self.EAR_CONSEC_FRAMES:
                    self.blink_history.append(cv2.getTickCount())
                    if len(self.blink_history) > 2:
                        self.blink_history.pop(0)
                    if len(self.blink_history) == 2:
                        time_diff = (self.blink_history[1] - self.blink_history[0]) / cv2.getTickFrequency()
                        if time_diff < 0.75:
                            self.current_sticker_index = (self.current_sticker_index + 1) % len(self.sticker_list)
                            print(f"切换贴纸至索引 {self.current_sticker_index}")
                self.blink_counter = 0

        # 显示图像

        self.display_image(frame)

    def display_image(self, frame):
        """在UI的QLabel上显示图像"""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        # OpenCV是BGR格式，QImage需要RGB格式
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        pixmap = QPixmap.fromImage(q_img)
        # 缩放图像以适应QLabel，并保持宽高比
        self.ui.qlabel.setPixmap(
            pixmap.scaled(
                self.ui.qlabel.width(),
                self.ui.qlabel.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )



    def update_current_time(self):
        """动态更新时间显示"""
        current_time = datetime.now().strftime("%Y年%m月%d日 %A %H:%M:%S")
        self.ui.datelabel.setText(current_time)
    def display_weather(self):
        """异步获取天气信息"""
        api_key = "0f45b923d11fe907232ecf601930588c"
        city_adcode = "440300"

        # 实时天气请求
        current_url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={city_adcode}&key={api_key}"
        current_request = QNetworkRequest(QtCore.QUrl(current_url))
        self.current_reply = self.network_manager.get(current_request)
        self.current_reply.finished.connect(self.handle_current_weather)

        # 天气预报请求
        forecast_url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={city_adcode}&key={api_key}&extensions=all"
        forecast_request = QNetworkRequest(QtCore.QUrl(forecast_url))
        self.forecast_reply = self.network_manager.get(forecast_request)
        self.forecast_reply.finished.connect(self.handle_forecast_weather)

    def handle_current_weather(self):
        """处理实时天气响应"""
        reply = self.sender()
        try:
            if reply.error() == QNetworkReply.NoError:
                data = json.loads(bytes(reply.readAll()).decode('utf-8'))
                if data["status"] == "1":
                    current = data["lives"][0]
                    # 使用纯文本格式
                    weather_text = (
                        f"实时天气\n"
                        f"天气: {current['weather']}\n"
                        f"温度: {current['temperature']}℃\n"
                        f"湿度: {current['humidity']}%\n"
                        f"风力: {current['winddirection']}风{current['windpower']}级"
                    )
                    self.ui.weatheredit.setText(weather_text)
                else:
                    self.ui.weatheredit.setText(f"实时天气获取失败: {data['info']}")
            else:
                self.ui.weatheredit.setText(f"网络错误: {reply.errorString()}")
        except Exception as e:
            self.ui.weatheredit.setText(f"处理实时天气出错: {str(e)}")
        finally:
            reply.deleteLater()

    def handle_forecast_weather(self):
        """处理天气预报响应"""
        reply = self.sender()
        try:
            if reply.error() == QNetworkReply.NoError:
                data = json.loads(bytes(reply.readAll()).decode('utf-8'))
                if data["status"] == "1" and "forecasts" in data:
                    forecast = data["forecasts"][0]


                    forecast_html = f"<h3>未来{len(forecast['casts'])}天预报</h3>"
                    forecast_html += "<table border='0' style='border-collapse: collapse; width: 100%; border: none;'>"
                    forecast_html += """
                    <tr>
                        <th style='padding: 10px 5px; border: none;'>日期</th>
                        <th style='padding: 10px 5px; border: none;'>白天</th>
                        <th style='padding: 10px 5px; border: none;'>夜间</th>
                        <th style='padding: 10px 5px; border: none;'>温度</th>
                        <th style='padding: 10px 5px; border: none;'>风力</th>
                    </tr>
                    """

                    for day in forecast["casts"]:
                        date_str = datetime.strptime(day['date'], "%Y-%m-%d").strftime("%m月%d日")
                        forecast_html += f"""
                        <tr style='border-bottom: 10px solid transparent;'>  <!-- 增加行间距 -->
                            <td style='padding: 10px 5px; border: none;'>{date_str}</td>
                            <td style='padding: 10px 5px; border: none;'>{day['dayweather']}</td>
                            <td style='padding: 10px 5px; border: none;'>{day['nightweather']}</td>
                            <td style='padding: 10px 5px; border: none;'>{day['nighttemp']}~{day['daytemp']}℃</td>
                            <td style='padding: 10px 5px; border: none;'>{day['daywind']}风{day['daypower']}级</td>
                        </tr>
                        """
                    forecast_html += "</table>"
                    self.ui.futureedit.setHtml(forecast_html)
                else:
                    self.ui.futureedit.setText(f"天气预报获取失败: {data.get('info', '无预报数据')}")
            else:
                self.ui.futureedit.setText(f"网络错误: {reply.errorString()}")
        except Exception as e:
            self.ui.futureedit.setText(f"处理天气预报出错: {str(e)}")
        finally:
            reply.deleteLater()


    def closeEvent(self, event):
        """关闭窗口时释放摄像头"""
        self.cap.release()
        event.accept()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = Camera()

    window.show()
    sys.exit(app.exec_())