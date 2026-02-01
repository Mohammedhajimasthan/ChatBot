import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found")

# Gemini config
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-flash-latest")

# FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    message: str

# Serve frontend
@app.get("/")
def home():
    return FileResponse("index.html")

# Chat API
@app.post("/chat")
async def chatbox(req: ChatRequest):
    try:
        chat = model.start_chat(history=[])
        response = chat.send_message(req.message)

        if not response or not response.text:
            raise ValueError("Empty response from model")

        return {"response": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))