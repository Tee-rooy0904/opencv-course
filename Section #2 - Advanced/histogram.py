#pylint:disable=no-member

import cv2 as cv
import matplotlib.pyplot as plt
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

    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv2_imshow(gray)

    mask = cv.circle(blank, (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)

    masked = cv.bitwise_and(img,img,mask=mask)
    print("Masked Image:")
    cv2_imshow(masked)

    # Grayscale histogram
    # gray_hist = cv.calcHist([gray], [0], mask, [256], [0,256] )

    # plt.figure()
    # plt.title('Grayscale Histogram')
    # plt.xlabel('Bins')
    # plt.ylabel('# of pixels')
    # plt.plot(gray_hist)
    # plt.xlim([0,256])
    # plt.show()

    # Colour Histogram
    plt.figure()
    plt.title('Colour Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    colors = ('b', 'g', 'r')
    for i,col in enumerate(colors):
        hist = cv.calcHist([img], [i], mask, [256], [0,256])
        plt.plot(hist, color=col)
        plt.xlim([0,256])

    plt.show()
