"""
llm.py — Strict document-QA via Groq API.

Sends the retrieved context + user question to a Groq-hosted LLM
with a system prompt that forbids hallucination.
"""

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Strict anti-hallucination prompt template
SYSTEM_PROMPT = (
    "You are a document question answering system.\n\n"
    "Only answer using the provided context.\n"
    "If the answer is not explicitly present in the context, "
    'reply exactly:\n"The answer is not present in the uploaded document."\n\n'
    "Do not infer, guess, or add outside knowledge."
)

SYSTEM_PROMPT_WITH_SYLLABUS = (
    "You are a document question answering system that is also constrained "
    "by a syllabus.\n\n"
    "Rules:\n"
    "1. Only answer using the provided context from the study notes.\n"
    "2. Only include information that falls within the syllabus topics listed below.\n"
    "3. If the answer is not explicitly present in the context, "
    'reply exactly:\n"The answer is not present in the uploaded document."\n'
    "4. If the answer is in the notes but NOT covered by the syllabus, "
    'reply exactly:\n"This topic is outside the scope of your syllabus."\n\n'
    "Do not infer, guess, or add outside knowledge."
)

USER_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer strictly using the context."
)

USER_TEMPLATE_WITH_SYLLABUS = (
    "Syllabus Topics:\n{syllabus}\n\n"
    "Context from Notes:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer strictly using the context, only if the topic is within the syllabus."
)

# Default model — fast and capable on Groq
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def ask_llm(
    context: str,
    question: str,
    syllabus_topics: str = "",
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Send the context + question to Groq and return the model's answer.

    If syllabus_topics is provided, the prompt constrains answers
    to only topics covered by the syllabus.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY environment variable is not set."

    client = Groq(api_key=api_key)

    if syllabus_topics:
        system = SYSTEM_PROMPT_WITH_SYLLABUS
        user_message = USER_TEMPLATE_WITH_SYLLABUS.format(
            syllabus=syllabus_topics, context=context, question=question
        )
    else:
        system = SYSTEM_PROMPT
        user_message = USER_TEMPLATE.format(context=context, question=question)

    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        temperature=0,       # deterministic — minimises hallucination
        max_tokens=1024,
    )

    return chat_completion.choices[0].message.content
