from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image


print("Loading TrOCR model...")

processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

print("Model loaded successfully!")

# Test with a handwritten line image
image = Image.open("handwritten_line.png").convert("RGB")

pixel_values = processor(
    images=image,
    return_tensors="pt"
).pixel_values

generated_ids = model.generate(
    pixel_values,
    max_new_tokens=64
)

text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print("\nRecognized text:")
print(text)