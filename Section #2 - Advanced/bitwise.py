# pylint:disable=no-member

import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow

blank = np.zeros((400, 400), dtype='uint8')

# Create a heart shape manually
heart = blank.copy()

t = np.linspace(0, 2 * np.pi, 1000)

x = 16 * np.sin(t) ** 3
y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

# Normalize and scale the heart to fit inside 400x400 image
x = (x - x.min()) / (x.max() - x.min())
y = (y - y.min()) / (y.max() - y.min())

x = (x * 300 + 50).astype(np.int32)
y = ((1 - y) * 280 + 60).astype(np.int32)

points = np.column_stack((x, y))

cv.fillPoly(heart, [points], 255)

# Create a circle
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)

print("Heart Shape:")
cv2_imshow(heart)

print("Circle Shape:")
cv2_imshow(circle)

# bitwise AND --> intersecting regions
bitwise_and = cv.bitwise_and(heart, circle)
print("Bitwise AND:")
cv2_imshow(bitwise_and)

# bitwise OR --> intersecting and non-intersecting regions
bitwise_or = cv.bitwise_or(heart, circle)
print("Bitwise OR:")
cv2_imshow(bitwise_or)

# bitwise XOR --> non-intersecting regions only
bitwise_xor = cv.bitwise_xor(heart, circle)
print("Bitwise XOR:")
cv2_imshow(bitwise_xor)

# bitwise NOT
bitwise_not = cv.bitwise_not(circle)
print("Circle NOT:")
cv2_imshow(bitwise_not)
