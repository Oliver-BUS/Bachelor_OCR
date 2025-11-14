import pytesseract
from PIL import Image

# Point exactly to your Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load a test image (make sure the file exists)
img = Image.open("test_image.png")

# Run OCR
text = pytesseract.image_to_string(img, lang="eng")

print("----- OCR RESULT -----")
print(text)
print("----------------------")

# print("Script started")

# import pytesseract
# from PIL import Image

# print("Imports OK")

# # Path to tesseract.exe – adjust ONLY if yours is different
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# print("Tesseract path set to:", pytesseract.pytesseract.tesseract_cmd)

# # Try to open the image
# try:
#     img = Image.open("test_image.png")
#     print("Image opened successfully")
# except Exception as e:
#     print("FAILED to open image:", e)
#     raise

# # Try OCR
# try:
#     text = pytesseract.image_to_string(img, lang="eng")
#     print("OCR call finished")
# except Exception as e:
#     print("FAILED during OCR:", e)
#     raise

# print("----- OCR RESULT -----")
# print(repr(text))
# print("----------------------")

