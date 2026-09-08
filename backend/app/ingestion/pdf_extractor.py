from io import BytesIO
from pypdf import PdfReader
from app.ingestion.ocr_utils import extract_text_with_ocr


def extract_pdf_text(file_content: bytes) -> dict:

    pdf_file = BytesIO(file_content)
    reader = PdfReader(pdf_file)

    pages = []

    # -----------------------------------
    # STEP 1: Normal PDF text extraction
    # -----------------------------------

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

    # Number of pages containing text
    non_empty_pages = sum(
        1
        for page in pages
        if page["text"]
    )

    # Percentage of pages containing text
    text_coverage = (
        non_empty_pages / len(pages)
        if pages
        else 0
    )

    # -----------------------------------
    # STEP 2: Detect scanned PDF
    # -----------------------------------

    needs_ocr = text_coverage < 0.5

    # -----------------------------------
    # STEP 3: Run OCR if required
    # -----------------------------------

    if needs_ocr:

        print("Scanned PDF detected. Starting OCR...")

        ocr_pages = extract_text_with_ocr(
            file_content
        )

        # Replace normal extraction with OCR result
        pages = [
            {
                "page": page_number,
                "text": text
            }
            for page_number, text
            in enumerate(ocr_pages, start=1)
        ]

        # Rebuild complete text
        full_text = "\n\n".join(
            page["text"]
            for page in pages
            if page["text"]
        )

        total_characters = len(full_text)

        # Recalculate coverage after OCR
        non_empty_pages = sum(
            1
            for page in pages
            if page["text"]
        )

        text_coverage = (
            non_empty_pages / len(pages)
            if pages
            else 0
        )

        print(
            f"OCR extracted {total_characters} characters."
        )

    # -----------------------------------
    # STEP 4: Return extracted content
    # -----------------------------------

    return {
        "text": full_text,
        "pages": pages,
        "page_count": len(reader.pages),
        "text_length": total_characters,
        "text_coverage": text_coverage,
        "needs_ocr": needs_ocr
    }