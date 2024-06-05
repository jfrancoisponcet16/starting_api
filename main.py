from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from typing import Dict
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from auth import authenticate
import ollama
import os
from dotenv import load_dotenv



app = FastAPI()




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
async def multimodal_endpoint(model: str = Form(...), prompt: str = Form(...), file: UploadFile = File(...), credentials: HTTPBasicCredentials = Depends(authenticate)):
    try:
        image_content = await file.read()
        encoded_image = base64.b64encode(image_content).decode()
        response = ollama.generate(model=model, prompt=prompt, images=[encoded_image])
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))