from app.ingestion.html_extractor import extract_html_text
from app.ingestion.pdf_extractor import extract_pdf_text


def extract_content(
    file_content: bytes,
    extension: str
):

    if extension == ".html":
        return {
            "format": "html",
            "text": extract_html_text(file_content)
        }

    if extension == ".pdf":
        return {
            "format": "pdf",
            **extract_pdf_text(file_content)
        }

    return {
        "format": "unknown",
        "text": ""
    }