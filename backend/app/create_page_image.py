import fitz


pdf = fitz.open("OB_2.pdf")

page = pdf[0]

pix = page.get_pixmap(
    matrix=fitz.Matrix(3, 3),
    alpha=False
)

pix.save("page1.png")

pdf.close()

print("page1.png created!")