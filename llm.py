"""llm.py - Lightweight Groq chat wrapper for the Streamlit chatbot."""

import os
import json

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful AI assistant in a web chat app. "
    "Give clear and concise answers."
)

# Default model — fast and capable on Groq
DEFAULT_MODEL = "llama-3.3-70b-versatile"


SPI_SYSTEM_PROMPT = (
    "You are an educational performance analyst. "
    "Given structured student profile data, provide practical, supportive, and specific guidance. "
    "Do not diagnose medical conditions. Avoid judgmental language."
)


def chat_with_llm(messages: list[dict], model: str = DEFAULT_MODEL) -> str:
    """Send chat history to Groq and return a single assistant response."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY environment variable is not set."

    client = Groq(api_key=api_key)

    payload = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in {"user", "assistant"} and isinstance(content, str):
            payload.append({"role": role, "content": content})

    chat_completion = client.chat.completions.create(
        model=model,
        messages=payload,
        temperature=0.7,
        max_tokens=1024,
    )

    return chat_completion.choices[0].message.content or "I couldn't generate a response."


def generate_spi_recommendation(student_profile: dict, model: str = DEFAULT_MODEL) -> str:
    """Generate a structured SPI recommendation from profile and quiz inputs."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY environment variable is not set."

    client = Groq(api_key=api_key)

    user_prompt = (
        "Analyze the following student profile and provide a concise report in markdown with these headings:\n"
        "1) Overall Performance Category (Excellent/Good/Average/Needs Improvement)\n"
        "2) Risk Level (Low/Medium/High)\n"
        "3) Key Strengths (3 bullets)\n"
        "4) Key Concerns (3 bullets)\n"
        "5) 4-Week Improvement Plan (weekly actions)\n"
        "6) Daily Study Routine (time-block style)\n"
        "7) Teacher/Mentor Interventions (3 bullets)\n"
        "8) Motivation Note (2-3 lines)\n\n"
        "Student Profile JSON:\n"
        f"{json.dumps(student_profile, indent=2)}"
    )

    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SPI_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=1200,
    )

    return chat_completion.choices[0].message.content or "I couldn't generate a recommendation."
