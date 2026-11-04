# pylint:disable=no-member

import cv2 as cv
from google.colab.patches import cv2_imshow

# Absolute paths inside mounted Google Drive
cascade_path = '/content/drive/MyDrive/IT QM/Section #3 - Faces/haar_face.xml'
img_path = '/content/drive/MyDrive/IT QM/Resources/Photos/Woody.jpg'

img = cv.imread(img_path)

if img is None:
    raise FileNotFoundError(f"Image was not loaded. Check this path: {img_path}")

haar_cascade = cv.CascadeClassifier(cascade_path)

if haar_cascade.empty():
    raise FileNotFoundError(f"Haar cascade was not loaded: {cascade_path}")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.equalizeHist(gray)

height, width = gray.shape

faces_rect = []

# ===============================
# MAIN FACE AREA
# ===============================

main_x1 = int(width * 0.05)
main_y1 = int(height * 0.30)
main_x2 = int(width * 0.75)
main_y2 = int(height * 0.95)

main_roi = gray[main_y1:main_y2, main_x1:main_x2]

main_faces = haar_cascade.detectMultiScale(
    main_roi,
    scaleFactor=1.04,
    minNeighbors=2,
    minSize=(70, 70),
    maxSize=(350, 350)
)

print("Main faces found:", len(main_faces))

if len(main_faces) > 0:
    main_face = max(main_faces, key=lambda face: face[2] * face[3])
    x, y, w, h = main_face
    faces_rect.append((x + main_x1, y + main_y1, w, h))


# ===============================
# TIKTOK PROFILE FACE AREA
# ===============================

profile_x1 = int(width * 0.75)
profile_y1 = int(height * 0.25)
profile_x2 = int(width * 0.98)
profile_y2 = int(height * 0.65)

profile_roi = gray[profile_y1:profile_y2, profile_x1:profile_x2]

profile_faces = haar_cascade.detectMultiScale(
    profile_roi,
    scaleFactor=1.04,
    minNeighbors=3,
    minSize=(18, 18),
    maxSize=(90, 90)
)

print("TikTok profile faces found:", len(profile_faces))

if len(profile_faces) > 0:
    profile_face = max(profile_faces, key=lambda face: face[2] * face[3])
    x, y, w, h = profile_face
    faces_rect.append((x + profile_x1, y + profile_y1, w, h))


# ===============================
# DRAW DETECTION + NAME
# ===============================

print("Total faces found:", len(faces_rect))

for (x, y, w, h) in faces_rect:
    name = "Woody"

    cv.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        thickness=2
    )

    cv.putText(
        img,
        name,
        (x, y - 10),
        cv.FONT_HERSHEY_COMPLEX,
        1.0,
        (0, 255, 0),
        thickness=2
    )

# Using Colab's native image rendering system
cv2_imshow(img)
