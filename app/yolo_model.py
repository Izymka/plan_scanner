from ultralytics import YOLO
from config import settings


class YOLOModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = settings["class_names"]  # Названия классов из конфигурации

    def predict(self, image):
        """
        Делает предсказание на изображении.
        
        :param image: изображение в формате numpy array (BGR)
        :return: результаты предсказания
        """
        results = self.model(image)
        return results

    def get_detections(self, results, image_shape):
        """
        Извлекает детекции из результатов предсказания.
        
        :param results: результаты предсказания YOLO
        :param image_shape: размеры изображения (height, width)
        :return: список детекций
        """
        detections = []
        height, width = image_shape[:2]

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Получаем координаты bbox
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Вычисляем центр bbox
                    x_center = int((x1 + x2) / 2)
                    y_center = int((y1 + y2) / 2)

                    # Получаем класс и уверенность
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())

                    # Получаем название класса
                    class_name = self.class_names[class_id] if class_id < len(
                        self.class_names) else f"unknown_{class_id}"

                    detection = {
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': {
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2)
                        },
                        'center': {
                            'x': x_center,
                            'y': y_center
                        }
                    }

                    detections.append(detection)

        return detections
