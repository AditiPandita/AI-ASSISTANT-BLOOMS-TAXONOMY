import os

from dotenv import load_dotenv
from google import genai


# Load environment variables
load_dotenv()


# Read Gemini configuration
api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)


# Validate API key
if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not set in the .env file."
    )


# Create Gemini client
client = genai.Client(
    api_key=api_key
)


def generate_answer(
    question: str,
    context: str
) -> str:

    if not question.strip():
        return "Please enter a question."

    if not context.strip():
        return (
            "I could not find relevant information "
            "in the uploaded document."
        )

    prompt = f"""
You are a reference-bounded academic assistant.

Your task is to answer the user's question using ONLY
the information provided in the context below.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not supported by the context, say:
   "I could not find the answer in the uploaded document."
4. Keep the answer clear and academically appropriate.
5. Mention the relevant page number when possible.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    if not response.text:
        return (
            "I could not generate an answer "
            "from the provided document."
        )

    return response.text.strip()