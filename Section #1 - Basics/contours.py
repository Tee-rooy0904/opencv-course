#pylint:disable=no-member

import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow

img = cv.imread('/content/drive/MyDrive/IT QM/Resources/Photos/cats.jpg')

# Check if image was loaded successfully
if img is None:
    print("Error: Could not load image. Please check the path and ensure the image exists.")
else:
    cv2_imshow(img)

    blank = np.zeros(img.shape, dtype='uint8')
    cv2_imshow(blank)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv2_imshow(gray)

    blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
    cv2_imshow(blur)

    canny = cv.Canny(blur, 125, 175)
    cv2_imshow(canny)

    # ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
    # cv2_imshow('Thresh', thresh)

    contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    print(f'{len(contours)} contour(s) found!')

    # Create a new blank image for drawing contours, ensuring it's the same size and type as img
    blank_for_contours = np.zeros(img.shape, dtype='uint8')
    cv.drawContours(blank_for_contours, contours, -1, (0,79,72), 1)
    cv2_imshow(blank_for_contours)
