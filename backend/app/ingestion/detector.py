from pathlib import Path


def detect_file_type(filename: str, content_type: str | None = None) -> dict:
    extension = Path(filename).suffix.lower()

    return {
        "filename": filename,
        "extension": extension,
        "content_type": content_type,
    }