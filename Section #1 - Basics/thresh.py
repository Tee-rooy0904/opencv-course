#pylint:disable=no-member

import cv2 as cv
from google.colab.patches import cv2_imshow

img = cv.imread('/content/drive/MyDrive/IT QM/Resources/Photos/Woody.jpg')

# Check if image was loaded successfully
if img is None:
    print("Error: Could not load image from '/content/drive/MyDrive/IT QM/Resources/Photos/Woody.jpg'. Please check the path.")
else:
    cv2_imshow(img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv2_imshow(gray)

    # Simple Thresholding
    threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY )
    cv2_imshow(thresh)

    threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV )
    cv2_imshow(thresh_inv)

    # Adaptive Thresholding
    adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 9)
    cv2_imshow(adaptive_thresh)

# cv.waitKey(0) is not needed for cv2_imshow
