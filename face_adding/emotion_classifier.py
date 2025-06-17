import cv2
import numpy as np
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input


def preprocess_face(face_image):
    # 如果face_image不是灰度图片，则转换为灰度图
    if len(face_image.shape) == 3:
        # 转换为灰度图
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    # 调整为模型输入尺寸
    resized = cv2.resize(gray, (48, 48))
    # 归一化处理
    preprocessed = preprocess_input(resized)
    return np.expand_dims(preprocessed, axis=0)


class EmotionClassifier:
    def __init__(self, model_path='models/emotion_model.h5'):
        # 加载预训练的表情识别模型
        # self.model = load_model(model_path)  # 取消注释可以用模型来识别表情
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.emotion_history = []
        self.history_size = 10
        self.weather_mapping = {
            'neutral': 'normal',
            'happy': 'sunny',
            'sad': 'light_rain',
            'surprise': 'thunderstorm',
            'angry': 'snow',
            'fear': 'thunder',
            'disgust': 'fog'
        }

    def predict_emotion(self, face_image):
        """预测人脸表情"""
        # 预处理
        preprocessed = preprocess_face(face_image)
        # 预测
        predictions = self.model.predict(preprocessed)[0]
        # 获取最高概率的表情
        emotion_idx = np.argmax(predictions)
        return self.emotion_labels[emotion_idx], predictions[emotion_idx]

    def update_emotion_history(self, emotion, confidence):
        """更新表情历史记录"""
        # 只记录高置信度的预测
        if confidence > 0.6:
            self.emotion_history.append(emotion)
            if len(self.emotion_history) > self.history_size:
                self.emotion_history.pop(0)

    def get_dominant_emotion(self):
        """获取主导表情（基于历史记录）"""
        if not self.emotion_history:
            return 'neutral'

        # 计算表情权重（最近的表情权重更高）
        weights = np.linspace(0.5, 1.0, len(self.emotion_history))
        emotion_counts = {}

        for i, emotion in enumerate(self.emotion_history):
            weight = weights[i]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + weight

        # 返回权重最高的表情
        return max(emotion_counts, key=emotion_counts.get)

    def get_weather_for_emotion(self, emotion):
        """获取表情对应的天气类型"""
        return self.weather_mapping.get(emotion, 'normal')
