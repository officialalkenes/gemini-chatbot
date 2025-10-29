"""
Simple Chatbot API using FastAPI, Gemini Flash 2.5, and LangChain
"""

import os
import json
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# Load environment variables
load_dotenv()

# Global variable to store the LLM
llm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global llm

    # Startup: Initialize the LLM
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",  # Gemini Flash 2.5
        google_api_key=api_key,
        temperature=0.7,
        max_tokens=1024,
    )
    print("✓ Gemini Flash 2.5 model initialized")

    yield

    # Shutdown: cleanup if needed
    print("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Gemini Chatbot API",
    description="A simple chatbot using Gemini Flash 2.5 and LangChain with streaming",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class Message(BaseModel):
    """Single message in a conversation"""
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User's message")
    conversation_history: Optional[List[Message]] = Field(
        default=None,
        description="Optional conversation history for context"
    )
    system_prompt: Optional[str] = Field(
        default="You are a helpful AI assistant.",
        description="System prompt to set the assistant's behavior"
    )


# Helper function to convert messages
def convert_to_langchain_messages(messages: List[Message], system_prompt: str):
    """Convert Pydantic messages to LangChain message objects"""
    lc_messages = [SystemMessage(content=system_prompt)]

    for msg in messages:
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            lc_messages.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            lc_messages.append(SystemMessage(content=msg.content))

    return lc_messages


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Gemini Chatbot API",
        "version": "0.1.0",
        "endpoints": {
            "POST /chat": "Send a message to the chatbot with streaming response",
            "GET /health": "Check API health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "gemini-2.0-flash-exp",
        "llm_initialized": llm is not None
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat with the Gemini model - Streaming Response

    Send a message and get a streaming response token by token.
    Optionally include conversation history for context.
    """
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="LLM not initialized. Please check your API key."
        )

    async def generate():
        try:
        
            conversation_history = request.conversation_history or []

            messages = convert_to_langchain_messages(
                conversation_history,
                request.system_prompt
            )

            # Add the current user message
            messages.append(HumanMessage(content=request.message))

            # Stream response from Gemini
            full_response = ""
            async for chunk in llm.astream(messages):
                if getattr(chunk, "content", None):
                    full_response += chunk.content
                    
                    yield f"data: {json.dumps({'content': chunk.content, 'done': False})}\n\n"

            # Send completion message with full response and updated history
            updated_history = conversation_history + [
                Message(role="user", content=request.message),
                Message(role="assistant", content=full_response)
            ]

            completion_data = {
                'content': '',
                'done': True,
                'full_response': full_response,
                'conversation_history': [msg.dict() for msg in updated_history]
            }

            yield f"data: {json.dumps(completion_data)}\n\n"

        except Exception as e:
            error_data = json.dumps({'error': str(e), 'done': True})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable buffering in nginx
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
