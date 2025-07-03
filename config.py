import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings():
    """Настройки приложения"""
    
    # Основные настройки
    app_name: str = "YOLO Детектор Дверей и Окон"
    app_version: str = "1.0.0"
    
    # Настройки сервера
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    # Настройки файлов
    max_file_size: int = 16 * 1024 * 1024  # 16MB
    allowed_extensions: set = {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif"}
    
    # Настройки модели
    model_path: str = "door_window_yolo11_detection.pt"
    class_names: list = ["door", "window"]
    
    # Настройки детекции
    confidence_threshold: float = 0.5
    colors: dict = {
        'door': (0, 255, 0),    # Зеленый
        'window': (255, 0, 0)   # Синий
    }

# Создаем экземпляр настроек
settings = {
    "app_name": os.getenv("APP_NAME", "YOLO Детектор Дверей и Окон"),
    "app_host": os.getenv("APP_HOST", "0.0.0.0"),
    "class_names": ["door", "window"],
    "max_file_size": 16 * 1024 * 1024
}