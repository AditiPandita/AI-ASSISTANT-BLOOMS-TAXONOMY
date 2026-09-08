import cv2
import numpy as np
from PIL import Image


def preprocess_line(image_path: str):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not load {image_path}"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Detect horizontal notebook lines
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (100, 1)
    )

    horizontal_lines = cv2.morphologyEx(
        gray,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    # Remove notebook ruling
    cleaned = cv2.subtract(
        gray,
        horizontal_lines
    )

    # Improve contrast
    cleaned = cv2.normalize(
        cleaned,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Threshold handwriting
    binary = cv2.threshold(
        cleaned,
        200,
        255,
        cv2.THRESH_BINARY
    )[1]

    # Find handwriting bounding box
    inverted = cv2.bitwise_not(binary)

    coords = cv2.findNonZero(inverted)

    if coords is None:
        return image_path

    x, y, w, h = cv2.boundingRect(coords)

    padding = 15

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(binary.shape[1], x + w + padding)
    y2 = min(binary.shape[0], y + h + padding)

    cropped = binary[
        y1:y2,
        x1:x2
    ]

    output_path = image_path.replace(
        ".png",
        "_clean.png"
    )

    cv2.imwrite(
        output_path,
        cropped
    )

    return output_path