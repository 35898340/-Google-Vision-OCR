from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from google.cloud import vision
from google.oauth2 import service_account

import os

app = FastAPI()

KEY_PATH = "vision_key.json"  # Временно храним файл рядом
credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
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
