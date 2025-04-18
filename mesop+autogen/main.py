import anthropic
import mesop as me
import mesop.labs as mel
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if debug mode is enabled
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

# Debug function to help diagnose issues
def debug_messages(messages):
    """Print message details for debugging"""
    for i, msg in enumerate(messages):
        content_preview = msg.get("content", "")[:30] + "..." if len(msg.get("content", "")) > 30 else msg.get("content", "")
        print(f"Message {i}: role={msg.get('role')}, content={content_preview}")
    return True

@me.page(
    security_policy=me.SecurityPolicy(
        allowed_iframe_parents=["https://google.github.io"]
    ),
    path="/",
    title="Mesop Demo Chat",
)
def page():
    mel.chat(transform, title="Anthropic Chat", bot_user="Claude")

def transform(input: str, history: list[mel.ChatMessage]):
    # Get API key from environment variables
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        yield "Error: ANTHROPIC_API_KEY not found in environment variables."
        return
    
    # Filter out messages with empty content and convert to Anthropic format
    anthropic_messages = []
    for message in history:
        if message.content.strip():  # Only include non-empty messages
            role = "user" if message.role == "user" else "assistant"
            anthropic_messages.append({"role": role, "content": message.content})
    
    # Add the current user message (which should never be empty)
    anthropic_messages.append({"role": "user", "content": input})
    
    # Print debug info if DEBUG is enabled
    if DEBUG:
        print(f"Number of messages: {len(anthropic_messages)}")
        debug_messages(anthropic_messages)
    
    try:
        # Initialize the Anthropic client
        client = anthropic.Anthropic(api_key=api_key)
        
        # Stream the response from Anthropic
        with client.messages.stream(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1024,
            system="You are a helpful AI assistant named Claude. Provide clear and accurate responses.",
            messages=anthropic_messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"Error: {str(e)}"

if __name__ == "__main__":
    me.run()
