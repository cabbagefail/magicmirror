from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from PyQt5.QtMultimediaWidgets import QCameraViewfinder
from PyQt5.QtMultimedia import QCamera
import sys
import json
from datetime import datetime
import camera
import menu


class Mycamera(QtWidgets.QMainWindow, camera.Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 初始化网络管理器
        self.network_manager = QNetworkAccessManager()

        # 设置定时器动态更新时间
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_current_time)
        self.timer.start(1000)  # 每秒更新一次

        # 初始化摄像头
        self.init_camera()

        # 获取天气信息
        self.display_weather()

    def init_camera(self):
        try:
            self.camer = QCamera()
            self.camer.setViewfinder(self.widget)
            self.camer.start()
        except Exception as e:
            print("摄像头初始化失败:", e)

    def update_current_time(self):
        current_time = datetime.now().strftime("%Y年%m月%d日 %A %H:%M:%S")
        self.label.setText(current_time)

    def display_weather(self):
        api_key = "0f45b923d11fe907232ecf601930588c"
        city_adcode = "310000"

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
        reply = self.sender()
        try:
            if reply.error() == QNetworkReply.NoError:
                data = json.loads(bytes(reply.readAll()).decode('utf-8'))
                if data["status"] == "1":
                    current = data["lives"][0]
                    current_html = f"""
                    <h3>实时天气</h3>
                    <p><b>天气:</b> {current['weather']}</p>
                    <p><b>温度:</b> {current['temperature']}℃</p>
                    <p><b>湿度:</b> {current['humidity']}%</p>
                    <p><b>风力:</b> {current['winddirection']}风{current['windpower']}级</p>
                    """
                    self.currentedit.setHtml(current_html)
                else:
                    self.currentedit.setText(f"实时天气获取失败: {data['info']}")
            else:
                self.currentedit.setText(f"网络错误: {reply.errorString()}")
        except Exception as e:
            self.currentedit.setText(f"处理实时天气出错: {str(e)}")
        finally:
            reply.deleteLater()

    def handle_forecast_weather(self):
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
                        <tr style='border-bottom: 10px solid transparent;'> 
                            <td style='padding: 10px 5px; border: none;'>{date_str}</td>
                            <td style='padding: 10px 5px; border: none;'>{day['dayweather']}</td>
                            <td style='padding: 10px 5px; border: none;'>{day['nightweather']}</td>
                            <td style='padding: 10px 5px; border: none;'>{day['nighttemp']}~{day['daytemp']}℃</td>
                            <td style='padding: 10px 5px; border: none;'>{day['daywind']}风{day['daypower']}级</td>
                        </tr>
                        """
                    forecast_html += "</table>"
                    self.foreedit.setHtml(forecast_html)
                else:
                    self.foreedit.setText(f"天气预报获取失败: {data.get('info', '无预报数据')}")
            else:
                self.foreedit.setText(f"网络错误: {reply.errorString()}")
        except Exception as e:
            self.foreedit.setText(f"处理天气预报出错: {str(e)}")
        finally:
            reply.deleteLater()

    def closeEvent(self, event):
        if hasattr(self, 'camer'):
            self.camer.stop()
        self.timer.stop()
        event.accept()


class Menu(QtWidgets.QMainWindow, menu.Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.to_the_camera)

    def to_the_camera(self):
        self.close()
        window2.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window1 = Menu()
    window2 = Mycamera()
    window1.show()
    sys.exit(app.exec_())