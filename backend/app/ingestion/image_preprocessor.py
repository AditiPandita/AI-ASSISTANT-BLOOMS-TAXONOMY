
import cv2
import numpy as np
import fitz


def preprocess_page_image(image: np.ndarray) -> dict:
    """
    Prepare a document page for OCR and visual analysis.

    Returns:
        original: Original image
        processed: Enhanced grayscale image
        binary: Adaptive threshold image
    """

    if image is None or image.size == 0:
        raise ValueError("Invalid page image.")

    # STEP 1: Preserve original image
    original = image.copy()

    # STEP 2: Correct minor rotation using text orientation
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    coordinates = np.column_stack(
        np.where(gray < 180)
    )

    if len(coordinates) > 100:

        angle = cv2.minAreaRect(
            coordinates.astype(np.float32)
        )[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Avoid applying unreliable large rotations
        if 0.5 < abs(angle) < 10:

            height, width = gray.shape

            center = (width // 2, height // 2)

            matrix = cv2.getRotationMatrix2D(
                center,
                angle,
                1.0
            )

            gray = cv2.warpAffine(
                gray,
                matrix,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

    # STEP 3: Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # STEP 4: Reduce noise
    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        10,
        7,
        21
    )

    # STEP 5: Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return {
        "original": original,
        "processed": denoised,
        "binary": binary
    }


def preprocess_pdf_pages(
    file_content: bytes,
    zoom: float = 2.5
) -> list[dict]:
    """
    Convert PDF pages into images and preprocess them.
    """

    pdf = fitz.open(
        stream=file_content,
        filetype="pdf"
    )

    results = []

    try:

        for page_number, page in enumerate(
            pdf,
            start=1
        ):

            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(zoom, zoom),
                alpha=False
            )

            image = np.frombuffer(
                pixmap.samples,
                dtype=np.uint8
            ).reshape(
                pixmap.height,
                pixmap.width,
                3
            )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )

            processed = preprocess_page_image(image)

            results.append({
                "page": page_number,
                **processed
            })

    finally:
        pdf.close()

    return results
