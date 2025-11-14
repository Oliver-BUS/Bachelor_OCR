from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en"
)

image_path = "test_image.png"   # file must exist

result = ocr.ocr(image_path, cls=True)

print("---- RESULTS ----")
for line in result:
    for box, (text, score) in line:
        print(f"{score:.3f}  {text}")
