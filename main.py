from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from google.cloud import vision
from google.oauth2 import service_account

import os
import json

app = FastAPI()

# Получаем переменную
json_str = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if not json_str:
    raise Exception("Переменная GOOGLE_APPLICATION_CREDENTIALS_JSON не установлена")

# Преобразуем строку обратно в словарь
service_account_info = json.loads(json_str)

# Ключ нужно расэкранировать
if "private_key" in service_account_info:
    service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")

# Создаём объект учётных данных
credentials = service_account.Credentials.from_service_account_info(service_account_info)
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
