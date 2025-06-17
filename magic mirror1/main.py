from PyQt5.QtWidgets import QMainWindow
from camera import Ui_widget  # 你的UI模块
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
from WeatherMirrorApp import SunnyEffects, SnowyEffects, RainyEffects, get_weather
import time

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
        self.weather = get_weather("Shenzhen")
        print(f"当前天气: {self.weather}")

        if not self.cap.isOpened():
            print("无法打开摄像头")
            sys.exit(-1)

        # 初始化特效
        self._reset_weather_effect()

        # 状态控制
        self.face_detection_enabled = False  # 默认关闭人脸识别
        self.ui.qbutton.setText("开启人脸识别")
        self.ui.qbutton.clicked.connect(self.toggle_face_mode)

        # 初始化参数
        self.init_stickers()
        self.init_detection_params()

        # 网络管理
        self.network_manager = QNetworkAccessManager(self)

        # 定时器设置
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 约30FPS

        self.time_timer = QTimer(self)
        self.time_timer.timeout.connect(self.update_current_time)
        self.time_timer.start(1000)  # 1秒更新一次时间

        self.weather_timer = QTimer(self)  # 天气更新(30分钟)
        self.weather_timer.timeout.connect(self.display_weather)
        self.weather_timer.start(1800000)

        # 性能监控
        self.last_time = time.time()
        self.frame_count = 0

        # 初始显示
        self.display_weather()
        self.update_current_time()

    def _reset_weather_effect(self):
        """根据当前天气重置特效"""
        if self.weather == 'rain':
            self.effect = RainyEffects(self.cap)
        elif self.weather == 'clouds':
            self.effect = RainyEffects(self.cap)
        elif self.weather == 'thunderrain':
            self.effect = RainyEffects(self.cap, True)
        elif self.weather == 'snow':
            self.effect = SnowyEffects(self.cap)
        else:
            self.effect = SunnyEffects(self.cap)
        print(f"已重置天气特效: {type(self.effect).__name__}")

    def toggle_face_mode(self):
        """切换人脸识别模式"""
        self.face_detection_enabled = not self.face_detection_enabled

        if self.face_detection_enabled:
            self.ui.qbutton.setText("关闭人脸识别")
            print("进入人脸识别模式：贴纸+动态天气")
        else:
            self.ui.qbutton.setText("开启人脸识别")
            self._reset_weather_effect()
            print("返回纯天气特效模式")

    def change_sticker(self, index):
        """切换贴纸"""
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
        """更新摄像头帧"""
        try:
            # 性能监控
            self.frame_count += 1
            if self.frame_count % 30 == 0:  # 每30帧计算一次FPS
                fps = 30 / (time.time() - self.last_time)
                self.last_time = time.time()

            ret, frame = self.cap.read()
            if not ret:
                return

            # 转换为RGB格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 应用特效
            if self.face_detection_enabled:
                frame = self._process_face_effects(frame)
            else:
                frame = self.effect.add_effects()  # 修改这里

            # 显示图像
            self.display_image(frame)

        except Exception as e:
            print(f"帧处理错误: {str(e)}")

    def _process_face_effects(self, frame):
        """处理人脸特效"""
        # 先应用天气特效
        frame = self.effect.add_effects()  # 修改这里

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = detector(gray, 0)

        for face in faces:
            landmarks = predictor(gray, face)

            # 应用贴纸
            apply_sticker(frame, landmarks, self.current_sticker_index,
                          self.sticker_list, self.scale_factors)

            # 动态天气切换
            self._change_effect_by_expression(landmarks)

            # 微笑检测激活胡须
            mouth_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
            mar = calculate_mar(mouth_points)

            if mar > self.MAR_THRESHOLD:
                self.smile_counter += 1
                if self.smile_counter >= self.MAR_CONSEC_FRAMES:
                    self.show_whiskers = True
                    self.last_smile_time = time.time()
                    self.smile_counter = 0
            else:
                self.smile_counter = 0
                if self.show_whiskers and (time.time() - self.last_smile_time > self.timeout_threshold):
                    self.show_whiskers = False

            if self.show_whiskers:
                apply_whiskers(frame, landmarks, self.whiskers_img, scale_factor=2.3)

        return frame

    def _change_effect_by_expression(self, landmarks):
        """根据表情切换天气特效"""
        mouth_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
        mar = calculate_mar(mouth_points)

        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

        # 表情判断
        if mar > 0.25:  # 大哭
            if not isinstance(self.effect, RainyEffects):
                self.effect = RainyEffects(self.cap)
                print("检测到大哭 → 雨天特效")
        elif ear > 0.3:  # 眼镜大张，惊讶
            if not isinstance(self.effect, SnowyEffects):
                self.effect = SnowyEffects(self.cap)
                print("检测到惊讶 → 雪天特效")
        else:  # 平静
            if not isinstance(self.effect, SunnyEffects):
                self.effect = SunnyEffects(self.cap)
                print("检测到平静 → 晴天特效")

    def display_image(self, frame):
        """显示图像"""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.ui.qlabel.setPixmap(
            pixmap.scaled(
                self.ui.qlabel.width(),
                self.ui.qlabel.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def update_current_time(self):
        """更新时间显示"""
        current_time = datetime.now().strftime("%Y年%m月%d日 %A %H:%M:%S")
        self.ui.datelabel.setText(current_time)

    def display_weather(self):
        """获取天气信息"""
        api_key = "0f45b923d11fe907232ecf601930588c"
        city_adcode = "440300"

        current_url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={city_adcode}&key={api_key}"
        current_request = QNetworkRequest(QtCore.QUrl(current_url))
        self.current_reply = self.network_manager.get(current_request)
        self.current_reply.finished.connect(self.handle_current_weather)

        forecast_url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={city_adcode}&key={api_key}&extensions=all"
        forecast_request = QNetworkRequest(QtCore.QUrl(forecast_url))
        self.forecast_reply = self.network_manager.get(forecast_request)
        self.forecast_reply.finished.connect(self.handle_forecast_weather)

    def handle_current_weather(self):
        """处理实时天气"""
        reply = self.sender()
        try:
            if reply.error() == QNetworkReply.NoError:
                data = json.loads(bytes(reply.readAll()).decode('utf-8'))
                if data["status"] == "1":
                    current = data["lives"][0]
                    weather_text = (
                        f"实时天气\n"
                        f"天气: {current['weather']}\n"
                        f"温度: {current['temperature']}℃\n"
                        f"湿度: {current['humidity']}%\n"
                        f"风力: {current['winddirection']}风{current['windpower']}级"
                    )
                    self.ui.weatheredit.setText(weather_text)
        finally:
            reply.deleteLater()

    def handle_forecast_weather(self):
        """处理天气预报"""
        reply = self.sender()
        try:
            if reply.error() == QNetworkReply.NoError:
                data = json.loads(bytes(reply.readAll()).decode('utf-8'))
                if data["status"] == "1" and "forecasts" in data:
                    forecast = data["forecasts"][0]
                    forecast_html = self._format_forecast_html(forecast)
                    self.ui.futureedit.setHtml(forecast_html)
        finally:
            reply.deleteLater()

    def _format_forecast_html(self, forecast):
        """格式化天气预报HTML"""
        html = f"<h3>未来{len(forecast['casts'])}天预报</h3>"
        html += "<table border='0' style='border-collapse: collapse; width: 100%;'>"
        html += """
        <tr>
            <th style='padding: 8px;'>日期</th>
            <th style='padding: 8px;'>白天</th>
            <th style='padding: 8px;'>夜间</th>
            <th style='padding: 8px;'>温度</th>
            <th style='padding: 8px;'>风力</th>
        </tr>
        """
        for day in forecast["casts"]:
            date_str = datetime.strptime(day['date'], "%Y-%m-%d").strftime("%m/%d")
            html += f"""
            <tr>
                <td style='padding: 8px;'>{date_str}</td>
                <td style='padding: 8px;'>{day['dayweather']}</td>
                <td style='padding: 8px;'>{day['nightweather']}</td>
                <td style='padding: 8px;'>{day['nighttemp']}~{day['daytemp']}℃</td>
                <td style='padding: 8px;'>{day['daywind']}风{day['daypower']}级</td>
            </tr>
            """
        html += "</table>"
        return html

    def closeEvent(self, event):
        """关闭事件"""
        self.cap.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Camera()
    window.show()
    sys.exit(app.exec_())