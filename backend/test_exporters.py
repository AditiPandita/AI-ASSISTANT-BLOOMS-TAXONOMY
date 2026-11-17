from app.exporters import generate_pdf, generate_docx

sample_questions = [
    {
        "question": "What is an Operating System?",
        "answer": "An Operating System manages computer hardware and software.",
        "bloom_level": "Remember",
        "marks": 5,
        "difficulty": "Easy",
        "source_pages": [1]
    },
    {
        "question": "Explain virtual memory.",
        "answer": "Virtual memory uses secondary storage as an extension of RAM.",
        "bloom_level": "Understand",
        "marks": 5,
        "difficulty": "Medium",
        "source_pages": [3]
    }
]

pdf_file = generate_pdf(
    sample_questions,
    include_answers=True,
    include_sources=True
)

with open("test_question_paper.pdf", "wb") as file:
    file.write(pdf_file.getvalue())


docx_file = generate_docx(
    sample_questions,
    include_answers=True,
    include_sources=True
)

with open("test_question_paper.docx", "wb") as file:
    file.write(docx_file.getvalue())


print("PDF and DOCX generated successfully!")