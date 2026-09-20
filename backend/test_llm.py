from app.llm import generate_answer


context = """
[Page 1]

The domicile certificate number is
DOM/0204/15Jul2020/200050871.
"""


question = (
    "What is the certificate number mentioned "
    "on the domicile certificate?"
)


answer = generate_answer(
    question,
    context
)


print("\n========== GEMINI ANSWER ==========\n")

print(answer)