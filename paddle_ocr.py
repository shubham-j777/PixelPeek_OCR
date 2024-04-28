from PIL import Image
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import layoutparser as lp
from paddleocr import PaddleOCR, draw_ocr
import tkinter as tk
from tkinter import filedialog

#select image

root = tk.Tk()
root.withdraw()

img_path = filedialog.askopenfilename()
image = cv2.imread(img_path)
#image = image[..., ::-1]

# Load model
model = lp.PaddleDetectionLayoutModel(config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",threshold=0.5,label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},enforce_cpu=False,enable_mkldnn=True)

ocr = PaddleOCR(lang='en',invert=True,image_orientation=True,binarize=True)# ,image_orientation=True,invert=True, binarize=True

# Process image and extract OCR results
image_cv = cv2.imread(img_path)
output = ocr.ocr(img_path)[0]

# Extract bounding boxes and texts
boxes = [line[0] for line in output]
texts = [line[1][0] for line in output]

# Output texts to a text file
output_file = 'outputs/detected_texts.txt'
with open(output_file, 'w') as f:
    for text in texts:
        f.write(text + '\n')

# Draw bounding boxes and texts on the image
image_boxes = image_cv.copy()
for box, text in zip(boxes, texts):
    # Calculate the width and height of the bounding box
    box_width = int(box[2][0]) - int(box[0][0])
    box_height = int(box[2][1]) - int(box[0][1])
    
    # Determine the font scale based on the bounding box size
    font_scale = min(box_width, box_height) / 50  # You can adjust the divisor to fit the text size
    
    # Draw the bounding box
    cv2.rectangle(image_boxes, (int(box[0][0]), int(box[0][1])), (int(box[2][0]), int(box[2][1])), (0, 0, 255), 1)
    
    # Draw the text
    cv2.putText(image_boxes, text, (int(box[0][0]), int(box[0][1])), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (222, 0, 0), 1)

# Save the image with bounding boxes and texts
cv2.imwrite('outputs/detections.jpg', image_boxes)
