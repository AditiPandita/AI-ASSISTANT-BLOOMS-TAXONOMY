from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from line_preprocessor import preprocess_line
from PIL import Image
import glob
import os
import torch


print("Loading TrOCR...")

processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

model.eval()

print("TrOCR loaded!")


files = sorted(
    glob.glob("detected_line_*.png"),
    key=lambda x: int(
        os.path.basename(x)
        .split("_")[2]
        .split(".")[0]
    )
)

print(f"\nProcessing {len(files)} lines...\n")


for index, file in enumerate(files[:5], start=1):

    clean_file = preprocess_line(file)

    image = Image.open(
        clean_file
    ).convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values

    with torch.no_grad():

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=64
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    print(f"LINE {index}")
    print(text)
    print("-" * 70)