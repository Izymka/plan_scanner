from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import cv2
import numpy as np
from PIL import Image
import io
import base64
from app.yolo_model import YOLOModel
from app.utils import process_image, allowed_file
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="YOLO Детектор Дверей и Окон", version="1.0.0")

# Монтируем статические файлы
app.mount("/static", 
          StaticFiles(directory=os.path.join(BASE_DIR, "static")), 
          name="static")

# Инициализация модели
model = YOLOModel('door_window_yolo11_detection.pt')

@app.get("/", response_class=HTMLResponse)
async def index():
    """Главная страница с HTML интерфейсом"""
    template_path = os.path.join(BASE_DIR, "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Загрузка и обработка изображения для детекции объектов
    """
    # Проверяем, что файл загружен
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не выбран")
    
    # Проверяем формат файла
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, 
                            detail="Неподдерживаемый формат файла")
    
    try:
        # Читаем изображение
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Конвертируем в RGB если необходимо
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Конвертируем в numpy array для OpenCV
        image_array = np.array(image)
        image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Делаем предсказание
        results = model.predict(image_cv)
        
        # Обрабатываем результаты
        processed_image, detections = process_image(image_cv, results)
        
        # Конвертируем обработанное изображение в base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "image": image_base64,
            "detections": detections,
            "total_objects": len(detections)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, 
                            detail=f"Ошибка обработки изображения: {str(e)}")

@app.get("/health")
async def health_check():
    """Проверка работоспособности API"""
    return {"status": "OK", "message": "API работает корректно"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)