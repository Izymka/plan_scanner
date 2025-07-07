import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Создаем экземпляр настроек
settings = {
    "app_name": os.getenv("APP_NAME", "YOLO Детектор Дверей и Окон"),
    "app_host": os.getenv("APP_HOST", "0.0.0.0"),
    "class_names": ["door", "window"],
    "max_file_size": 16 * 1024 * 1024,
    "allowed_extensions": ("png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif"),
    "reload": True,
    "confidence_threshold": os.getenv("CONF_THRESHOLD", 0.6),
    "app_version": os.getenv("APP_VERSION", "1.0.0"),
    "port": os.getenv("PORT", 8000),
    "model_path": os.getenv("MODEL_PATH", "door_window_yolo11_detection.pt"),
     "colors": {
        "door": (0, 255, 0),     # Зеленый
        "window": (255, 0, 0)    # Синий
    }

}