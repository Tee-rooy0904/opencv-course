#pylint:disable=no-member

import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow

# Using the absolute path to your Google Drive image
img_path = '/content/drive/MyDrive/IT QM/Resources/Photos/Woody.jpg'
img = cv.imread(img_path)

if img is None:
    print(f"Error: Could not load image from '{img_path}'. Please check the path.")
else:
    print("Original Image:")
    cv2_imshow(img)

    blank = np.zeros(img.shape[:2], dtype='uint8')
    print("Blank Image:")
    cv2_imshow(blank)

    circle = cv.circle(blank.copy(), (img.shape[1]//2 + 45,img.shape[0]//2), 100, 255, -1)

    rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)

    weird_shape = cv.bitwise_and(circle,rectangle)
    print("Weird Shape Mask:")
    cv2_imshow(weird_shape)

    masked = cv.bitwise_and(img,img,mask=weird_shape)
    print("Weird Shaped Masked Image:")
    cv2_imshow(masked)
