try:
    import Image
except ImportError:
   from PIL import Image
import pytesseract
print(pytesseract.image_to_string(Image.open('test.jpg')))

 #print(pytesseract.image_to_string(Image.open('test-european.jpg'), lang='fra'))