import cv2
from PIL import Image
import pytesseract
import tkinter as tk
from tkinter import filedialog

#select image

root = tk.Tk()
root.withdraw()

img_path = filedialog.askopenfilename()
image = cv2.imread(img_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

# Use pytesseract to perform OCR
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(gray, lang='eng', config=custom_config)

# Output texts to a text file
output_file = 'outputs/detected_texts.txt'
with open(output_file, 'w') as f:
    f.write(text)

# Draw bounding boxes and texts on the image
detections_image = image.copy()
detections = pytesseract.image_to_data(gray, lang='eng', output_type=pytesseract.Output.DICT)
num_boxes = len(detections['text'])
for i in range(num_boxes):
    x, y, w, h = detections['left'][i], detections['top'][i], detections['width'][i], detections['height'][i]
    conf = int(detections['conf'][i])
    if conf > 0:
        cv2.rectangle(detections_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(detections_image, detections['text'][i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the image with bounding boxes and texts
cv2.imwrite('outputs/detections.jpg', detections_image)
