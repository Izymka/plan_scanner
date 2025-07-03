import cv2
import numpy as np
import os
from config import settings

def allowed_file(filename):
    """
    Проверяет, является ли файл допустимым изображением.
    
    :param filename: имя файла
    :return: True если файл поддерживается, False иначе
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in settings.allowed_extensions

def process_image(image, results):
    """
    Обрабатывает изображение, добавляя bbox и возвращая информацию о детекциях.
    
    :param image: исходное изображение в формате numpy array (BGR)
    :param results: результаты предсказания YOLO
    :return: кортеж (обработанное_изображение, список_детекций)
    """
    # Создаем копию изображения для рисования
    processed_image = image.copy()
    height, width = image.shape[:2]
    
    detections = []
    class_names = settings.class_names
    
    # Цвета для разных классов из конфигурации
    colors = settings.colors
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Получаем координаты bbox
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Вычисляем центр bbox
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                
                # Получаем класс и уверенность
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                
                # Получаем название класса
                class_name = class_names[class_id] if class_id < len(class_names) else f"unknown_{class_id}"
                
                # Получаем цвет для класса
                color = colors.get(class_name, (0, 0, 255))  # Красный по умолчанию
                
                # Рисуем bbox
                cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
                
                # Рисуем центр
                cv2.circle(processed_image, (x_center, y_center), 5, color, -1)
                
                # Добавляем текст с классом и уверенностью
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Рисуем фон для текста
                cv2.rectangle(processed_image, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), 
                            color, -1)
                
                # Рисуем текст
                cv2.putText(processed_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Добавляем координаты центра рядом с центром
                center_label = f"({x_center}, {y_center})"
                cv2.putText(processed_image, center_label, 
                           (x_center + 10, y_center - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Сохраняем информацию о детекции
                detection = {
                    'class': class_name,
                    'confidence': round(confidence, 3),
                    'bbox': {
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2
                    },
                    'center': {
                        'x': x_center,
                        'y': y_center
                    }
                }
                
                detections.append(detection)
    
    return processed_image, detections

def get_supported_image_extensions():
    """
    Возвращает список поддерживаемых расширений изображений.
    """
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']