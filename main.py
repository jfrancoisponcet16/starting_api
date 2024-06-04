from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from auth import authenticate
import ollama
import os
from dotenv import load_dotenv



app = FastAPI()

# @app.get("/api/chat")
# async def proxy_chat(query: str, credentials: HTTPBasicCredentials = Depends(authenticate)):
#     url = "http://127.0.0.1:xxxxx/api/chat"
#     async with httpx.AsyncClient() as client:
#         try:
#             response = await client.get(url, params={"query": query})
#             response.raise_for_status()
#             return JSONResponse(content=response.json())
#         except httpx.HTTPStatusError as exc:
#             raise HTTPException(status_code=exc.response.status_code, detail="Error from Ollama API")
#         except httpx.RequestError as exc:
#             raise HTTPException(status_code=500, detail="Error contacting Ollama API")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



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



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)