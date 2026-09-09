def chunk_pages(
    pages: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> list[dict]:

    chunks = []

    chunk_id = 1

    for page in pages:

        page_number = page.get("page")
        text = page.get("text", "").strip()

        if not text:
            continue

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page_number,
                    "text": chunk_text
                })

                chunk_id += 1

            start = end - chunk_overlap

    return chunks