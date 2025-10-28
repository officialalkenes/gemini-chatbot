"""
Test client for the Gemini Chatbot API
"""
import httpx
import asyncio
import json


async def test_simple_chat():
    """Test the simple chat endpoint"""
    print("\n=== Testing Simple Chat ===")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/chat/simple",
            params={"message": "Hello! What can you help me with?"}
        )
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"User: {result['message']}")
        print(f"Assistant: {result['response']}")


async def test_chat_with_history():
    """Test the main chat endpoint with conversation history"""
    print("\n=== Testing Chat with History ===")
    
    # First message
    conversation_history = []
    
    messages = [
        "Hi! My name is Alice.",
        "What's my name?",
        "Can you write a short poem about AI?"
    ]
    
    async with httpx.AsyncClient() as client:
        for msg in messages:
            print(f"\nUser: {msg}")
            
            response = await client.post(
                "http://localhost:8000/chat",
                json={
                    "message": msg,
                    "conversation_history": conversation_history,
                    "system_prompt": "You are a friendly and helpful AI assistant."
                }
            )
            
            result = response.json()
            print(f"Assistant: {result['response']}")
            
            # Update conversation history for next message
            conversation_history = result['conversation_history']


async def test_health():
    """Test the health check endpoint"""
    print("\n=== Testing Health Check ===")
    
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")


async def main():
    """Run all tests"""
    print("Gemini Chatbot API Test Client")
    print("=" * 50)
    
    try:
        await test_health()
        await test_simple_chat()
        await test_chat_with_history()
        
    except httpx.ConnectError:
        print("\n❌ Error: Could not connect to the API.")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())