import io
import fitz
import pytesseract

from PIL import Image, ImageOps, ImageFilter


def extract_text_with_ocr(file_content: bytes) -> list[str]:

    pages_text = []

    pdf = fitz.open(
        stream=file_content,
        filetype="pdf"
    )

    print(f"OCR started. Total pages: {len(pdf)}")

    for page_number, page in enumerate(pdf, start=1):

        print(f"OCR processing page {page_number}...")

        # Render page at higher resolution
        pix = page.get_pixmap(
            matrix=fitz.Matrix(4, 4),
            alpha=False
        )

        image_bytes = pix.tobytes("png")

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        print(
            f"Original image size: "
            f"{image.width} x {image.height}"
        )

        # Convert to grayscale
        image = ImageOps.grayscale(image)

        # Improve contrast
        image = ImageOps.autocontrast(image)

        # Slight sharpening
        image = image.filter(
            ImageFilter.SHARPEN
        )

        # OCR
        text = pytesseract.image_to_string(
            image,
            lang="eng",
            config="--oem 3 --psm 3"
        )

        text = text.strip()

        print(
            f"Extracted characters: {len(text)}"
        )

        pages_text.append(text)

    pdf.close()

    print("OCR completed.")

    return pages_text