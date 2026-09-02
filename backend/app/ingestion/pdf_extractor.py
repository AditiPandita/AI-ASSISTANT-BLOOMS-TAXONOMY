from io import BytesIO
from pypdf import PdfReader


def extract_pdf_text(file_content: bytes) -> dict:
    pdf_file = BytesIO(file_content)
    reader = PdfReader(pdf_file)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append({
            "page": page_number,
            "text": text.strip()
        })

    full_text = "\n\n".join(
        page["text"]
        for page in pages
        if page["text"]
    )

    total_characters = len(full_text)

    # Number of pages from which we extracted some text
    non_empty_pages = sum(
        1
        for page in pages
        if page["text"]
    )

    # Percentage of pages containing extracted text
    text_coverage = (
        non_empty_pages / len(pages)
        if pages
        else 0
    )

    # Temporary OCR decision rule
    needs_ocr = text_coverage < 0.5

    return {
        "text": full_text,
        "pages": pages,
        "page_count": len(reader.pages),
        "text_length": total_characters,
         "text_coverage": text_coverage,
        "needs_ocr": needs_ocr
    }