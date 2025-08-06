from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.chatbot import generate_response
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/api")
def chat(request: ChatRequest):
    try:
        response = generate_response(request.messages)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Untuk run: python main.py
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
