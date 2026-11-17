
import json
import re
import time

from google.genai import types

from app.llm import client, model_name


# --------------------------------------------------
# BLOOM'S TAXONOMY LEVELS
# --------------------------------------------------

BLOOM_LEVELS = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create"
]


# --------------------------------------------------
# GENERATE BLOOM QUESTIONS
# --------------------------------------------------

def generate_bloom_questions(
    context: str,
    bloom_levels: list[str],
    questions_per_level: int = 5,
    marks_per_question: float = 2
) -> list[dict]:

    # ----------------------------------------------
    # STEP 1: INPUT VALIDATION
    # ----------------------------------------------

    if not context.strip():
        raise ValueError("Document context cannot be empty.")

    if not bloom_levels:
        raise ValueError("At least one Bloom level is required.")

    invalid_levels = [
        level
        for level in bloom_levels
        if level not in BLOOM_LEVELS
    ]

    if invalid_levels:
        raise ValueError(
            f"Invalid Bloom levels: {invalid_levels}"
        )

    if not 1 <= questions_per_level <= 20:
        raise ValueError(
            "Questions per level must be between 1 and 20."
        )

    if marks_per_question <= 0:
        raise ValueError(
            "Marks per question must be greater than zero."
        )

    # ----------------------------------------------
    # STEP 2: PREPARE GEMINI PROMPT
    # ----------------------------------------------

    prompt = f"""
You are an expert academic assessment question generator.

Generate questions STRICTLY from the supplied document context.

BLOOM'S TAXONOMY LEVELS:

- Remember: Recall facts, terms and definitions.
- Understand: Explain, summarize and interpret concepts.
- Apply: Use concepts to solve problems or situations.
- Analyze: Examine relationships, compare and differentiate.
- Evaluate: Judge, justify and critically assess.
- Create: Design, formulate or develop something new using
  concepts supported by the document.

SELECTED BLOOM LEVELS:
{json.dumps(bloom_levels)}

NUMBER OF QUESTIONS PER LEVEL:
{questions_per_level}

MARKS PER QUESTION:
{marks_per_question}

STRICT RULES:

1. Use only information supported by the supplied context.
2. Do not invent facts, concepts, examples, names, dates or URLs.
3. Every question must genuinely match its assigned Bloom level.
4. Avoid duplicate or nearly identical questions.
5. Questions must be clear and academically appropriate.
6. Provide a concise expected answer for every question.
7. Assign exactly the requested marks to each question.
8. Return exactly the requested number of questions per level.
9. Each context section contains a Chunk ID and Page number.
10. For every question, return only the chunk IDs that directly
    support that question and its answer.
11. Never invent a Chunk ID.
12. Do not cite chunks that do not support the question.
13. If multiple chunks support a question, include all relevant IDs.
14. Do not include markdown or explanations outside the JSON.
15. Return valid JSON only.

DOCUMENT CONTEXT:
{context}

Return a JSON array with this exact structure:

[
    {{
        "question": "Question text",
        "answer": "Expected answer",
        "bloom_level": "Remember",
        "marks": {marks_per_question},
        "difficulty": "Easy",
        "source_chunk_ids": [1]
    }}
]
"""

    # ----------------------------------------------
    # STEP 3: CALL GEMINI WITH AUTOMATIC RETRIES
    # ----------------------------------------------

    response = None

    for attempt in range(3):

        try:

            print(
                f"Calling Gemini API "
                f"(attempt {attempt + 1}/3)..."
            )

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3
                )
            )

            break

        except Exception as error:

            error_message = str(error).upper()

            temporary_error = (
                "503" in error_message
                or "UNAVAILABLE" in error_message
                or "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
            )

            if not temporary_error:
                raise

            if attempt == 2:

                raise RuntimeError(
                    "Gemini API is temporarily unavailable "
                    "or rate-limited. Please try again later."
                ) from error

            wait_time = 2 ** (attempt + 1)

            print(
                f"Gemini temporarily unavailable or rate-limited. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

    # ----------------------------------------------
    # STEP 4: VALIDATE GEMINI RESPONSE
    # ----------------------------------------------

    if response is None or not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    try:

        questions = json.loads(response.text)

    except json.JSONDecodeError as error:

        raise RuntimeError(
            "Gemini returned invalid JSON."
        ) from error

    if not isinstance(questions, list):
        raise RuntimeError(
            "Gemini response must be a JSON array."
        )

    # ----------------------------------------------
    # STEP 5: VALIDATE QUESTION COUNT
    # ----------------------------------------------

    expected_count = (
        len(bloom_levels) * questions_per_level
    )

    if len(questions) != expected_count:

        raise RuntimeError(
            f"Expected {expected_count} questions, "
            f"but received {len(questions)}."
        )

    # ----------------------------------------------
    # STEP 6: EXTRACT VALID SOURCE CHUNK IDS
    # ----------------------------------------------

    valid_chunk_ids = {
        int(chunk_id)
        for chunk_id in re.findall(
            r"\[Chunk ID: (\d+) \| Page:",
            context
        )
    }

    if not valid_chunk_ids:
        raise RuntimeError(
            "No valid source chunk IDs found in context."
        )

    # ----------------------------------------------
    # STEP 7: VALIDATE EACH GENERATED QUESTION
    # ----------------------------------------------

    level_counts = {
        level: 0
        for level in bloom_levels
    }

    for question in questions:

        if not isinstance(question, dict):
            raise RuntimeError(
                "Each generated question must be a JSON object."
            )

        if question.get("bloom_level") not in bloom_levels:
            raise RuntimeError(
                "Gemini returned an unexpected Bloom level."
            )

        level_counts[question["bloom_level"]] += 1

        if not question.get("question"):
            raise RuntimeError(
                "A generated question is empty."
            )

        if not question.get("answer"):
            raise RuntimeError(
                "A generated answer is empty."
            )

        # ------------------------------------------
        # VALIDATE SOURCE CHUNK IDS
        # ------------------------------------------

        source_ids = question.get("source_chunk_ids")

        if not isinstance(source_ids, list) or not source_ids:
            raise RuntimeError(
                "A generated question has no source chunk IDs."
            )

        if not all(
            isinstance(chunk_id, int)
            and not isinstance(chunk_id, bool)
            for chunk_id in source_ids
        ):
            raise RuntimeError(
                "Source chunk IDs must be integers."
            )

        if len(source_ids) != len(set(source_ids)):
            raise RuntimeError(
                "Duplicate source chunk IDs were returned."
            )

        if not set(source_ids).issubset(valid_chunk_ids):
            raise RuntimeError(
                "Gemini returned invalid source chunk IDs."
            )

        question["source_chunk_ids"] = source_ids

        # Use the requested marks
        question["marks"] = marks_per_question

        # Validate difficulty
        if question.get("difficulty") not in [
            "Easy",
            "Medium",
            "Hard"
        ]:
            question["difficulty"] = "Medium"

    # ----------------------------------------------
    # STEP 8: VERIFY QUESTIONS PER BLOOM LEVEL
    # ----------------------------------------------

    for level, count in level_counts.items():

        if count != questions_per_level:

            raise RuntimeError(
                f"Expected {questions_per_level} questions "
                f"for {level}, but received {count}."
            )

    print(
        f"Successfully generated {len(questions)} "
        f"Bloom questions."
    )

    return questions
