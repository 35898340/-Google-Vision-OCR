from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from google.cloud import vision
from google.oauth2 import service_account

import os
import json

app = FastAPI()

# Создание временного ключа из переменной окружения
GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")

if not GOOGLE_CREDENTIALS_JSON:
    raise Exception("Переменная GOOGLE_APPLICATION_CREDENTIALS_JSON не установлена")

# Пишем во временный файл
KEY_FILE_PATH = "vision_key.json"
with open(KEY_FILE_PATH, "w") as f:
    f.write(GOOGLE_CREDENTIALS_JSON)

# Загружаем ключ и создаём клиента Vision API
credentials = service_account.Credentials.from_service_account_file(KEY_FILE_PATH)
client = vision.ImageAnnotatorClient(credentials=credentials)

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = vision.Image(content=contents)
        response = client.text_detection(image=image)

        if response.error.message:
            return JSONResponse(status_code=500, content={"error": response.error.message})

        texts = response.text_annotations
        if not texts:
            return {"text": ""}

        return {"text": texts[0].description.strip()}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
