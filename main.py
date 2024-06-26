from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Request
from typing import Dict
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from auth import authenticate
import ollama
from fastapi.encoders import jsonable_encoder
import torch
from transformers import AutoModel, AutoTokenizer
from fastapi.responses import JSONResponse
import io
from PIL import Image
import traceback
import easyocr
from doctr import DocumentFile
from doctr.models import ocr_predictor





app = FastAPI()

# Initialize the reader
reader = easyocr.Reader(['en'], gpu=True)
# Initialize the predictor once at startup
predictor = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)




class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    
class EmbeddingsRequest(BaseModel):
    model: str
    prompt: str
    
 


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, credentials: HTTPBasicCredentials = Depends(authenticate)):
    try:
        response = ollama.chat(model=request.model, messages=[message.model_dump() for message in request.messages])
        return response['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/api/embeddings")
async def embeddings_endpoint(request: EmbeddingsRequest, credentials: HTTPBasicCredentials = Depends(authenticate)):
    try:
        response = ollama.embeddings(model=request.model, prompt=request.prompt)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import base64


@app.post("/api/multimodal")
async def multimodal_endpoint(model: str = Form(...), prompt: str = Form(...), options: str = Form(...), file: UploadFile = File(...), credentials: HTTPBasicCredentials = Depends(authenticate)):
    try:
        image_content = await file.read()
        encoded_image = base64.b64encode(image_content).decode()
        response = ollama.generate(model=model, prompt=prompt, images=[encoded_image], options=options)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/extract_text/")
async def extract_text(request: Request, file: UploadFile = File(...), credentials: HTTPBasicCredentials = Depends(authenticate)):

    # Read the file contents
    contents = await file.read()

    # Process the image with OCR
    try:
        doc = DocumentFile.from_images([contents])
        doc = predictor(doc)
        text = doc.render()
    except Exception as e:
        return JSONResponse(content=jsonable_encoder({"error": str(e)}), status_code=500)
    finally:
        # Clean up resources
        del contents, doc
        gc.collect()

    # Return the extracted text
    return JSONResponse(content=jsonable_encoder({"text": text}))