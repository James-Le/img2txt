# Many thanks to wxwwt, I borrow a lot of code from https://github.com/wxwwt/opencv-picture-to-excel.
import os
import cv2
import re
import argparse
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from cnocr import CnOcr

# If you put tesseract.exe somewhere else, please re-define the following path. 
# pytesseract.pytesseract.tesseract_cmd = r'C://Program Files//Tesseract-OCR//tesseract.exe'
ocr = CnOcr()

parser = argparse.ArgumentParser(description='This is a Python3 script for OCR.')
parser.add_argument('src_path', metavar='src_path', type=str, help='The path of image file.')
parser.add_argument('tar_path', metavar='tar_path', type=str, help='The path of output file.')
args = parser.parse_args()

raw = cv2.imread(os.path.normpath(args.src_path), 1)
gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
rows, cols = binary.shape
scale = 12

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
eroded = cv2.erode(binary, kernel, iterations=1)
dilated_col = cv2.dilate(eroded, kernel, iterations=1)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
eroded = cv2.erode(binary, kernel, iterations=1)
dilated_row = cv2.dilate(eroded, kernel, iterations=1)

bitwise_and = cv2.bitwise_and(dilated_col, dilated_row)

merge = cv2.add(dilated_col, dilated_row)
merge2 = cv2.subtract(binary, merge)
merge3 = cv2.add(merge2, bitwise_and)

ys, xs = np.where(bitwise_and > 0)
y_point_arr, x_point_arr = [], []

i = 0
sort_x_point = np.sort(xs)
for i in range(len(sort_x_point) - 1):
    if sort_x_point[i+1] - sort_x_point[i] > 10:
        x_point_arr.append(sort_x_point[i])
    i += 1
x_point_arr.append(sort_x_point[i])

i = 0
sort_y_point = np.sort(ys)
for i in range(len(sort_y_point) - 1):
    if sort_y_point[i+1] - sort_y_point[i] > 10:
        y_point_arr.append(sort_y_point[i])
    i += 1
y_point_arr.append(sort_y_point[i])

data = [[] for i in range(len(y_point_arr)-1)]

for i in range(len(y_point_arr) - 1):
    for j in range(len(x_point_arr) - 1):
        cell = raw[y_point_arr[i]:y_point_arr[i+1], x_point_arr[j]:x_point_arr[j+1]]
        text = pytesseract.image_to_string(cell).strip()
        if not text:
            text = "".join(ocr.ocr_for_single_line(cell))
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[\x0c\n]', '', text)
        text = "".join(text)
        data[i].append(text)
        j += 1
    i += 1

data = pd.DataFrame(data)
data.to_csv(os.path.normpath(args.tar_path), encoding="utf-8", header=False, index=False)

data = None
ocr = None

print("All recognition has been completed!")