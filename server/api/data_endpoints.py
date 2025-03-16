from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from model.run_vlm import run_vlm
import os
import logging


data_router = APIRouter()


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Chat API - Handling VLM Requests
@data_router.post("/chat")
async def chat(files: list[UploadFile]):
    """ Receive uploaded image files and call the VLM model for inference """
    folder_path = './data/in'
    os.makedirs(folder_path, exist_ok=True)

    if not files:
        return Response(status_code=400, content="Error: No files received")

    for file in files:
        contents = await file.read()

        file_path = os.path.join(folder_path, file.filename)
        with open(file_path, 'wb') as f:
            f.write(contents)
        logger.info(f"Saved image: {file.filename} under {folder_path}")

    run_vlm()

    with open(results_file, 'r', encoding='utf-8') as file:
        results = json.load(file)

    return JSONResponse(results)


