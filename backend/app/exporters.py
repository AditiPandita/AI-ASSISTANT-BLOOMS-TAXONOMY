
from io import BytesIO
from xml.sax.saxutils import escape

from docx import Document as WordDocument
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    KeepTogether,
    HRFlowable
)


# ==========================================
# COMMON HELPERS
# ==========================================

def get_exam_details(exam_details, questions):

    details = exam_details or {}

    total_marks = sum(
        float(question.get("marks") or 0)
        for question in questions
    )

    return {
        "institution": details.get(
            "institution",
            "VISHWAKARMA INSTITUTE OF TECHNOLOGY"
        ),
        "exam_title": details.get(
            "exam_title",
            "EXAMINATION"
        ),
        "subject_name": details.get(
            "subject_name",
            "General Subject"
        ),
        "subject_code": details.get(
            "subject_code",
            "N/A"
        ),
        "duration": details.get(
            "duration",
            "2 Hours"
        ),
        "exam_date": details.get(
            "exam_date",
            "N/A"
        ),
        "instructions": details.get(
            "instructions",
            "Answer all questions."
        ),
        "total_questions": len(questions),
        "total_marks": total_marks
    }


def group_questions_by_bloom(questions):

    groups = {}

    for question in questions:

        level = question.get("bloom_level") or "General"

        if level not in groups:
            groups[level] = []

        groups[level].append(question)

    return groups


def get_section_title(level):

    return f"Section - {level}"


# ==========================================
# DOCX EXPORT
# ==========================================

def generate_docx(
    questions: list[dict],
    include_answers: bool = False,
    include_sources: bool = False,
    title: str = "AI Generated Question Paper",
    exam_details: dict | None = None,
    group_by_bloom: bool = False
) -> BytesIO:

    document = WordDocument()

    # ----------------------------------
    # PAGE FORMATTING
    # ----------------------------------

    section = document.sections[0]

    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

    # Default font
    normal_style = document.styles["Normal"]
    normal_style.font.name = "Arial"
    normal_style.font.size = Pt(10)

    details = get_exam_details(
        exam_details,
        questions
    )

    # ----------------------------------
    # EXAMINATION HEADER
    # ----------------------------------

    heading = document.add_heading(
        details["institution"],
        level=0
    )

    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER

    exam_heading = document.add_heading(
        details["exam_title"],
        level=1
    )

    exam_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subject_heading = document.add_paragraph()

    subject_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subject_run = subject_heading.add_run(
        details["subject_name"]
    )

    subject_run.bold = True
    subject_run.font.size = Pt(13)

    # ----------------------------------
    # EXAMINATION DETAILS
    # ----------------------------------

    document.add_paragraph(
        f"Subject Code: {details['subject_code']}"
    )

    document.add_paragraph(
        f"Date: {details['exam_date']}"
    )

    document.add_paragraph(
        f"Duration: {details['duration']}"
    )

    document.add_paragraph(
        f"Maximum Marks: {details['total_marks']:g}"
    )

    document.add_paragraph(
        f"Total Questions: {details['total_questions']}"
    )

    document.add_paragraph("")

    document.add_heading(
        "Instructions",
        level=2
    )

    document.add_paragraph(
        details["instructions"]
    )

    document.add_paragraph("")

    # ----------------------------------
    # QUESTION GENERATION
    # ----------------------------------

    question_number = 1

    if group_by_bloom:

        question_groups = group_questions_by_bloom(
            questions
        )

        for level, level_questions in question_groups.items():

            section_heading = document.add_heading(
                get_section_title(level),
                level=2
            )

            for question in level_questions:

                paragraph = document.add_paragraph()

                run = paragraph.add_run(
                    f"Q{question_number}. "
                    f"{question.get('question', '')}"
                )

                run.bold = True

                paragraph.add_run(
                    f" [{question.get('marks', 0)} Marks]"
                )

                if include_sources:

                    pages = question.get("source_pages") or []

                    document.add_paragraph(
                        f"Source Pages: "
                        f"{', '.join(map(str, pages)) or 'N/A'}"
                    )

                document.add_paragraph("")

                question_number += 1

    else:

        for question in questions:

            paragraph = document.add_paragraph()

            run = paragraph.add_run(
                f"Q{question_number}. "
                f"{question.get('question', '')}"
            )

            run.bold = True

            paragraph.add_run(
                f" [{question.get('marks', 0)} Marks]"
            )

            metadata = document.add_paragraph(
                f"Bloom Level: {question.get('bloom_level', 'N/A')} | "
                f"Difficulty: {question.get('difficulty', 'N/A')}"
            )

            metadata.runs[0].italic = True

            if include_sources:

                pages = question.get("source_pages") or []

                document.add_paragraph(
                    f"Source Pages: "
                    f"{', '.join(map(str, pages)) or 'N/A'}"
                )

            document.add_paragraph("")

            question_number += 1

    # ----------------------------------
    # ANSWER KEY
    # ----------------------------------

    if include_answers:

        document.add_page_break()

        answer_heading = document.add_heading(
            "ANSWER KEY",
            level=0
        )

        answer_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER

        for index, question in enumerate(questions, start=1):

            document.add_heading(
                f"Answer {index}",
                level=2
            )

            document.add_paragraph(
                question.get("answer") or "Answer unavailable."
            )

            if include_sources:

                pages = question.get("source_pages") or []

                document.add_paragraph(
                    f"Source Pages: "
                    f"{', '.join(map(str, pages)) or 'N/A'}"
                )

    # ----------------------------------
    # SAVE DOCX
    # ----------------------------------

    output = BytesIO()

    document.save(output)

    output.seek(0)

    return output


# ==========================================
# PDF EXPORT
# ==========================================

def generate_pdf(
    questions: list[dict],
    include_answers: bool = False,
    include_sources: bool = False,
    title: str = "AI Generated Question Paper",
    exam_details: dict | None = None,
    group_by_bloom: bool = False
) -> BytesIO:

    output = BytesIO()

    pdf = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch
    )

    details = get_exam_details(
        exam_details,
        questions
    )

    styles = getSampleStyleSheet()

    # ----------------------------------
    # PDF STYLES
    # ----------------------------------

    institution_style = ParagraphStyle(
        "Institution",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=16,
        leading=20,
        spaceAfter=8
    )

    exam_title_style = ParagraphStyle(
        "ExamTitle",
        parent=styles["Heading2"],
        alignment=TA_CENTER,
        fontSize=13,
        leading=17,
        spaceAfter=8
    )

    subject_style = ParagraphStyle(
        "Subject",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=12,
        leading=16,
        spaceAfter=14
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        spaceBefore=15,
        spaceAfter=10
    )

    question_style = ParagraphStyle(
        "Question",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=16,
        spaceAfter=7
    )

    metadata_style = ParagraphStyle(
        "Metadata",
        parent=styles["Normal"],
        fontSize=8.5,
        textColor=colors.grey,
        spaceAfter=8
    )

    answer_style = ParagraphStyle(
        "Answer",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        spaceAfter=12
    )

    instruction_style = ParagraphStyle(
        "Instructions",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=14,
        spaceAfter=8
    )

    story = []

    # ----------------------------------
    # EXAMINATION HEADER
    # ----------------------------------

    story.append(
        Paragraph(
            escape(details["institution"]),
            institution_style
        )
    )

    story.append(
        Paragraph(
            escape(details["exam_title"]),
            exam_title_style
        )
    )

    story.append(
        Paragraph(
            f"<b>{escape(details['subject_name'])}</b>",
            subject_style
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.black
        )
    )

    story.append(Spacer(1, 12))

    # ----------------------------------
    # EXAMINATION INFORMATION
    # ----------------------------------

    information = [
        f"<b>Subject Code:</b> {escape(str(details['subject_code']))}",
        f"<b>Date:</b> {escape(str(details['exam_date']))}",
        f"<b>Duration:</b> {escape(str(details['duration']))}",
        f"<b>Maximum Marks:</b> {details['total_marks']:g}",
        f"<b>Total Questions:</b> {details['total_questions']}"
    ]

    for item in information:

        story.append(
            Paragraph(item, styles["Normal"])
        )

    story.append(Spacer(1, 12))

    # ----------------------------------
    # INSTRUCTIONS
    # ----------------------------------

    story.append(
        Paragraph(
            "<b>Instructions:</b>",
            styles["Heading3"]
        )
    )

    story.append(
        Paragraph(
            escape(details["instructions"]),
            instruction_style
        )
    )

    story.append(Spacer(1, 12))

    # ----------------------------------
    # QUESTION HELPER
    # ----------------------------------

    def create_question_block(question, number):

        question_text = escape(
            str(question.get("question") or "")
        )

        marks = question.get("marks") or 0

        block = [
            Paragraph(
                f"<b>Q{number}. {question_text}</b> "
                f"[{marks} Marks]",
                question_style
            )
        ]

        if not group_by_bloom:

            metadata = (
                f"Bloom Level: "
                f"{escape(str(question.get('bloom_level') or 'N/A'))}"
                f" | Difficulty: "
                f"{escape(str(question.get('difficulty') or 'N/A'))}"
            )

            block.append(
                Paragraph(metadata, metadata_style)
            )

        if include_sources:

            pages = question.get("source_pages") or []

            source_text = escape(
                ", ".join(map(str, pages)) or "N/A"
            )

            block.append(
                Paragraph(
                    f"Source Pages: {source_text}",
                    metadata_style
                )
            )

        block.append(Spacer(1, 10))

        return KeepTogether(block)

    # ----------------------------------
    # QUESTION PAPER
    # ----------------------------------

    question_number = 1

    if group_by_bloom:

        question_groups = group_questions_by_bloom(
            questions
        )

        for level, level_questions in question_groups.items():

            story.append(
                Paragraph(
                    escape(get_section_title(level)),
                    section_style
                )
            )

            for question in level_questions:

                story.append(
                    create_question_block(
                        question,
                        question_number
                    )
                )

                question_number += 1

    else:

        for question in questions:

            story.append(
                create_question_block(
                    question,
                    question_number
                )
            )

            question_number += 1

    # ----------------------------------
    # ANSWER KEY
    # ----------------------------------

    if include_answers:

        story.append(PageBreak())

        story.append(
            Paragraph(
                "ANSWER KEY",
                styles["Heading1"]
            )
        )

        story.append(Spacer(1, 12))

        for index, question in enumerate(questions, start=1):

            answer = escape(
                str(
                    question.get("answer")
                    or "Answer unavailable."
                )
            )

            story.append(
                Paragraph(
                    f"<b>Answer {index}</b>",
                    styles["Heading3"]
                )
            )

            story.append(
                Paragraph(
                    answer,
                    answer_style
                )
            )

            if include_sources:

                pages = question.get("source_pages") or []

                source_text = escape(
                    ", ".join(map(str, pages)) or "N/A"
                )

                story.append(
                    Paragraph(
                        f"Source Pages: {source_text}",
                        metadata_style
                    )
                )

    # ----------------------------------
    # BUILD PDF
    # ----------------------------------

    pdf.build(story)

    output.seek(0)

    return output
