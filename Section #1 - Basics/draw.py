#pylint:disable=no-member

import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow

blank = np.zeros((500,500,3), dtype='uint8')
cv2_imshow(blank)

# 1. Paint the image a certain colour
blank[200:300, 300:400] = 0,0,255
cv2_imshow(blank)

# 2. Draw a Rectangle
cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,255,0), thickness=-1)
cv2_imshow(blank)

# 3. Draw A circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0,0,255), thickness=-1)
cv2_imshow(blank)

# 4. Draw a line
cv.line(blank, (100,250), (300,400), (255,255,255), thickness=3)
cv2_imshow(blank)

# 5. Write text
cv.putText(blank, 'Hello, my name is Troy Budoy!!!', (0,225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,255), 2)
cv2_imshow(blank)
