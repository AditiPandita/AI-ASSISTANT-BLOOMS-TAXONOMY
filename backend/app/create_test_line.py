import fitz
from PIL import Image
import io

pdf_path = "OB_2.pdf"

pdf = fitz.open(pdf_path)

page = pdf[0]

# Render page at high resolution
pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)

image = Image.open(
    io.BytesIO(pix.tobytes("png"))
)

print("Rendered size:", image.size)

# Crop one handwritten line
# Coordinates are proportional to the first page
width, height = image.size

crop = image.crop((
    int(width * 0.08),
    int(height * 0.14),
    int(width * 0.93),
    int(height * 0.22)
))

crop.save("handwritten_line.png")
print("handwritten_line.png created!")

pdf.close()