import cv2
import numpy as np
from PIL import Image


def detect_handwriting_lines(image_path: str):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image could not be loaded.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect dark pixels
    binary = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY_INV
    )[1]

    # Detect horizontal notebook ruling lines
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (200, 1)
    )

    ruling_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    # Get Y positions of notebook lines
    projection = np.sum(
        ruling_lines > 0,
        axis=1
    )

    threshold = image.shape[1] * 0.08

    y_positions = np.where(
        projection > threshold
    )[0]

    # Group nearby Y positions
    grouped = []

    if len(y_positions) > 0:

        start = y_positions[0]
        previous = y_positions[0]

        for y in y_positions[1:]:

            if y - previous <= 3:
                previous = y
            else:
                grouped.append(
                    (start + previous) // 2
                )

                start = y
                previous = y

        grouped.append(
            (start + previous) // 2
        )

    # Remove very close duplicate rulings
    ruling_positions = []

    for y in grouped:

        if not ruling_positions:
            ruling_positions.append(y)

        elif y - ruling_positions[-1] > 15:
            ruling_positions.append(y)

    pil_image = Image.open(
        image_path
    ).convert("RGB")

    detected_lines = []

    # Create regions between notebook rulings
    for index in range(
        len(ruling_positions) - 1
    ):

        top = ruling_positions[index]
        bottom = ruling_positions[index + 1]

        # Ignore unusually large/small spaces
        height = bottom - top

        if height < 20 or height > 100:
            continue

        # Leave small margins around the ruling
        crop_top = top + 3
        crop_bottom = bottom - 2

        # Check whether this region contains handwriting
        region = binary[
            crop_top:crop_bottom,
            :
        ]

        ink_pixels = np.count_nonzero(region)

        region_area = region.shape[0] * region.shape[1]

        ink_ratio = (
            ink_pixels / region_area
            if region_area > 0
            else 0
        )

        # Ignore mostly empty notebook rows
        if ink_ratio < 0.01:
            continue

        left = int(
            pil_image.width * 0.06
        )

        right = int(
            pil_image.width * 0.96
        )

        crop = pil_image.crop(
            (
                left,
                crop_top,
                right,
                crop_bottom
            )
        )

        output_path = (
            f"detected_line_{len(detected_lines) + 1}.png"
        )

        crop.save(output_path)

        detected_lines.append(
            output_path
        )

    return detected_lines