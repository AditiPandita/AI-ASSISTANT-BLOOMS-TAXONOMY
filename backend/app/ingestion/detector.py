
from pathlib import Path


SUPPORTED_FORMATS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".html": "html",
    ".htm": "html"
}


def detect_file_type(
    filename: str,
    content_type: str | None = None
) -> dict:

    extension = Path(filename).suffix.lower()

    return {
        "filename": filename,
        "extension": extension,
        "content_type": content_type,
        "format": SUPPORTED_FORMATS.get(
            extension,
            "unknown"
        ),
        "supported": extension in SUPPORTED_FORMATS
    }
