from PIL import Image, ImageDraw
import glob
import os
import math


files = sorted(
    glob.glob("detected_line_*.png"),
    key=lambda x: int(
        os.path.basename(x)
        .split("_")[2]
        .split(".")[0]
    )
)

images = [
    Image.open(file).convert("RGB")
    for file in files
]

thumb_width = 900
thumbs = []

for i, image in enumerate(images, start=1):

    ratio = thumb_width / image.width
    height = int(image.height * ratio)

    image = image.resize(
        (thumb_width, height)
    )

    canvas = Image.new(
        "RGB",
        (thumb_width, height + 35),
        "white"
    )

    canvas.paste(image, (0, 35))

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (10, 8),
        f"Line {i}",
        fill="black"
    )

    thumbs.append(canvas)


columns = 1
rows = len(thumbs)

total_height = sum(
    image.height
    for image in thumbs
)

contact = Image.new(
    "RGB",
    (thumb_width, total_height),
    "white"
)

y = 0

for image in thumbs:

    contact.paste(
        image,
        (0, y)
    )

    y += image.height


contact.save(
    "detected_lines_contact_sheet.png"
)

print(
    f"Contact sheet created with {len(files)} lines."
)