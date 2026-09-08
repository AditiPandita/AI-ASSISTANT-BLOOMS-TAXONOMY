from handwriting_lines import detect_handwriting_lines


image_path = "page1.png"

lines = detect_handwriting_lines(image_path)

print("\nDetected lines:")

for line in lines:
    print(line)

print(f"\nTotal lines detected: {len(lines)}")