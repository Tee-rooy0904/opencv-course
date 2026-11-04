#pylint:disable=no-member

import cv2 as cv
from google.colab.patches import cv2_imshow

# Correct path pointing to your Google Drive location
img_path = '/content/drive/MyDrive/IT QM/Resources/Photos/Woody.jpg'
img = cv.imread(img_path)

if img is None:
    print(f"Error: Could not load image from '{img_path}'. Please verify the path.")
else:
    print("Original image:")
    cv2_imshow(img)

    # Averaging
    average = cv.blur(img, (3,3))
    print("Average Blur:")
    cv2_imshow(average)

    # Gaussian Blur
    gauss = cv.GaussianBlur(img, (3,3), 0)
    print("Gaussian Blur:")
    cv2_imshow(gauss)

    # Median Blur
    median = cv.medianBlur(img, 3)
    print("Median Blur:")
    cv2_imshow(median)

    # Bilateral Filter
    bilateral = cv.bilateralFilter(img, 10, 35, 25)
    print("Bilateral Filter:")
    cv2_imshow(bilateral)
