
from io import BytesIO
from docx import Document


def extract_docx_text(file_content: bytes) -> dict:
    """
    Extract text from DOCX paragraphs and tables.
    Return content in the same page-based structure
    expected by the existing upload pipeline.
    """

    document = Document(BytesIO(file_content))

    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    table_text = []

    for table in document.tables:
        for row in table.rows:

            cells = [
                cell.text.strip().replace("\n", " ")
                for cell in row.cells
            ]

            row_text = " | ".join(
                cell for cell in cells if cell
            )

            if row_text:
                table_text.append(row_text)

    full_text = "\n".join(paragraphs)

    if table_text:
        full_text += "\n\n" + "\n".join(table_text)

    # DOCX does not expose reliable rendered page
    # boundaries through python-docx.
    # Store its extracted content as one logical page.
    pages = []

    if full_text.strip():
        pages.append({
            "page": 1,
            "text": full_text.strip()
        })

    return {
        "text": full_text,
        "pages": pages,
        "page_count": len(pages),
        "paragraph_count": len(paragraphs),
        "table_count": len(document.tables),
        "text_length": len(full_text)
    }
