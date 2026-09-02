from bs4 import BeautifulSoup


def extract_html_text(file_content: bytes) -> str:
    soup = BeautifulSoup(file_content, "html.parser")

    # Remove content that is not useful as academic text
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")#converts to plain text

    # Clean empty lines and unnecessary spaces
    lines = []

    for line in text.splitlines():
        line = line.strip()

        if line:
            lines.append(line)#removes empty lines.

    return "\n".join(lines)
