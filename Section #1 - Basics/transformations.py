import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow

# Correcting the file path to the mounted Google Drive directory
img = cv.imread('/content/drive/MyDrive/IT QM/Resources/Photos/Woody.jpg')

if img is None:
    print("Error: Could not load image. Please verify the file path.")
else:
    print("Original Image:")
    cv2_imshow(img)

    # Translation
    def translate(img, x, y):
        transMat = np.float32([[1,0,x],[0,1,y]])
        dimensions = (img.shape[1], img.shape[0])
        return cv.warpAffine(img, transMat, dimensions)

    # -x --> Left
    # -y --> Up
    # x --> Right
    # y --> Down

    translated = translate(img, -100, 100)
    print("Translated Image:")
    cv2_imshow(translated)

    # Rotation
    def rotate(img, angle, rotPoint=None):
        (height,width) = img.shape[:2]

        if rotPoint is None:
            rotPoint = (width//2,height//2)

        rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
        dimensions = (width,height)

        return cv.warpAffine(img, rotMat, dimensions)

    rotated = rotate(img, -45)
    print("Rotated Image (-45 deg):")
    cv2_imshow(rotated)

    rotated_rotated = rotate(img, -90)
    print("Rotated Image (-90 deg):")
    cv2_imshow(rotated_rotated)

    # Resizing
    resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
    print("Resized Image:")
    cv2_imshow(resized)

    # Flipping
    flip = cv.flip(img, -1)
    print("Flipped Image:")
    cv2_imshow(flip)

    # Cropping
    cropped = img[200:400, 300:400]
    print("Cropped Image:")
    cv2_imshow(cropped)
