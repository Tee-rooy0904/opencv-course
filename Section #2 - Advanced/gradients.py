#pylint:disable=no-member

import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow

# Correcting path to mounted Google Drive folder
img_path = '/content/drive/MyDrive/IT QM/Resources/Photos/Woody.jpg'
img = cv.imread(img_path)

if img is None:
    print(f"Error: Could not load image from '{img_path}'. Please check the path.")
else:
    print("Original Image:")
    cv2_imshow(img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print("Gray Image:")
    cv2_imshow(gray)

    # Laplacian
    lap = cv.Laplacian(gray, cv.CV_64F)
    lap = np.uint8(np.absolute(lap))
    print("Laplacian:")
    cv2_imshow(lap)

    # Sobel 
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
    combined_sobel = cv.bitwise_or(sobelx, sobely)

    print("Sobel X:")
    cv2_imshow(sobelx)
    print("Sobel Y:")
    cv2_imshow(sobely)
    print("Combined Sobel:")
    cv2_imshow(combined_sobel)

    canny = cv.Canny(gray, 150, 175)
    print("Canny:")
    cv2_imshow(canny)
