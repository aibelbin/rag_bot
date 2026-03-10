"""
llm.py — Strict document-QA via Groq API.

Sends the retrieved context + user question to a Groq-hosted LLM
with a system prompt that forbids hallucination.
"""

import os
from groq import Groq

# Strict anti-hallucination prompt template
SYSTEM_PROMPT = (
    "You are a document question answering system.\n\n"
    "Only answer using the provided context.\n"
    "If the answer is not explicitly present in the context, "
    'reply exactly:\n"The answer is not present in the uploaded document."\n\n'
    "Do not infer, guess, or add outside knowledge."
)

USER_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer strictly using the context."
)

# Default model — fast and capable on Groq
DEFAULT_MODEL = "llama3-70b-8192"


def ask_llm(
    context: str,
    question: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Send the context + question to Groq and return the model's answer.

    Requires the GROQ_API_KEY environment variable to be set.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY environment variable is not set."

    client = Groq(api_key=api_key)

    user_message = USER_TEMPLATE.format(context=context, question=question)

    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,       # deterministic — minimises hallucination
        max_tokens=1024,
    )

    return chat_completion.choices[0].message.content
