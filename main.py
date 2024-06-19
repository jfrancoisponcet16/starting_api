from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from typing import Dict
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from auth import authenticate
import ollama
import os
from dotenv import load_dotenv
import torch
from transformers import AutoModel, AutoTokenizer
from fastapi.responses import JSONResponse
import io
from PIL import Image
import traceback





app = FastAPI()



# Load model
model_path = 'openbmb/MiniCPM-Llama3-V-2_5'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'int4' in model_path and device == 'mps':
    raise ValueError('Error: running int4 model with bitsandbytes this machine is not supported right now.')

model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(dtype=torch.float16).to(device=device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.eval()

ERROR_MSG = "Error, please retry"
model_name = 'MiniCPM-Llama3-V 2.5'



class MercuriumParams(BaseModel):
    params_form: str = "Sampling"
    num_beams: int = 3
    repetition_penalty: float = 1.2
    repetition_penalty_2: float = 1.05
    top_p: float = 0.8
    top_k: int = 100
    temperature: float = 0.7
    message: str



class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    
class EmbeddingsRequest(BaseModel):
    model: str
    prompt: str
    
 
@app.post("/Mercurium_generate/")
async def mercurium_generate(
    file: UploadFile = File(...), 
    message: str = Form(...),
    params_form: str = Form("Sampling"),
    num_beams: int = Form(3),
    repetition_penalty: float = Form(1.2),
    repetition_penalty_2: float = Form(1.05),
    top_p: float = Form(0.8),
    top_k: int = Form(100),
    temperature: float = Form(0.7),
    credentials: HTTPBasicCredentials = Depends(authenticate)
):
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Create message context
        msgs = [{"role": "user", "content": message}]
        
        # Set default chat parameters
        chat_params = {
            "stream": False,
            "sampling": False,
            "num_beams": 3,
            "repetition_penalty": 1.2,
            "max_new_tokens": 1024
        }
        
        # Update chat parameters based on the form input
        if params_form == 'Beam Search':
            chat_params.update({
                'sampling': False,
                'num_beams': num_beams,
                'repetition_penalty': repetition_penalty
            })
        else:
            chat_params.update({
                'sampling': True,
                'top_p': top_p,
                'top_k': top_k,
                'temperature': temperature,
                'repetition_penalty': repetition_penalty_2
            })
        
        # Generate response
        answer = model.chat(image=image, msgs=msgs, tokenizer=tokenizer, **chat_params)
        response = ''.join(answer)  # Assuming answer is iterable
        
        return JSONResponse(content={"response": response})
    
    except Exception as e:
        print(e)
        traceback.print_exc()
        return JSONResponse(content={"status": ERROR_MSG}, status_code=500)   
    
    

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
    


