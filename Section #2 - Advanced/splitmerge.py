#pylint:disable=no-member

import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow

# Using verified Google Drive path for Woody 2
img_path = '/content/drive/MyDrive/IT QM/Resources/Photos/Woody 2.jpg'
img = cv.imread(img_path)

if img is None:
    print(f"Error: Could not load image from '{img_path}'. Please verify that Woody 2.jpg exists in this directory.")
else:
    print("Original Image:")
    cv2_imshow(img)

    blank = np.zeros(img.shape[:2], dtype='uint8')

    b,g,r = cv.split(img)

    blue = cv.merge([b,blank,blank])
    green = cv.merge([blank,g,blank])
    red = cv.merge([blank,blank,r])

    print("Blue Channel Only:")
    cv2_imshow(blue)

    print("Green Channel Only:")
    cv2_imshow(green)

    print("Red Channel Only:")
    cv2_imshow(red)

    print(f"Image Shape: {img.shape}")
    print(f"B Channel Shape: {b.shape}")
    print(f"G Channel Shape: {g.shape}")
    print(f"R Channel Shape: {r.shape}")

    merged = cv.merge([b,g,r])
    print("Re-merged Image:")
    cv2_imshow(merged)
