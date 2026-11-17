
from pathlib import Path

import cv2

from app.ingestion.image_preprocessor import preprocess_pdf_pages


# Keep AIPYQ1.pdf inside the backend folder
PDF_PATH = Path("AIPYQ1.pdf")

OUTPUT_DIR = Path("preprocessing_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            "AIPYQ1.pdf not found. "
            "Place the PDF inside the backend folder."
        )

    file_content = PDF_PATH.read_bytes()

    print("Starting image preprocessing...")

    pages = preprocess_pdf_pages(
        file_content,
        zoom=2.5
    )

    for page in pages:

        page_number = page["page"]

        cv2.imwrite(
            str(OUTPUT_DIR / f"page_{page_number}_original.png"),
            page["original"]
        )

        cv2.imwrite(
            str(OUTPUT_DIR / f"page_{page_number}_processed.png"),
            page["processed"]
        )

        cv2.imwrite(
            str(OUTPUT_DIR / f"page_{page_number}_binary.png"),
            page["binary"]
        )

        print(
            f"Page {page_number} processed successfully."
        )

    print("\nPreprocessing completed!")
    print(f"Images saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
