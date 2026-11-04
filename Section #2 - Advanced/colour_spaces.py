#pylint:disable=no-member

import cv2 as cv
from google.colab.patches import cv2_imshow

# Correcting path to mounted Google Drive folder
img_path = '/content/drive/MyDrive/IT QM/Resources/Photos/Woody.jpg'
img = cv.imread(img_path)

if img is None:
    print(f"Error: Could not load image from '{img_path}'. Please check the path.")
else:
    print("Original Image:")
    cv2_imshow(img)

    # BGR to Grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print("Gray Image:")
    cv2_imshow(gray)

    # BGR to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    print("HSV Image:")
    cv2_imshow(hsv)

    # BGR to L*a*b
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    print("LAB Image:")
    cv2_imshow(lab)

    # BGR to RGB
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    print("RGB Image:")
    cv2_imshow(rgb)

    # L*a*b to BGR
    lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    print("LAB --> BGR Image:")
    cv2_imshow(lab_bgr)
