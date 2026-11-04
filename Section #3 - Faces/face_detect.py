# pylint:disable=no-member

import cv2 as cv
import os
from google.colab.patches import cv2_imshow

# Dynamic Search to find where 'haar_face.xml' is stored in your Google Drive
def find_file(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

print("Searching for 'haar_face.xml' in Google Drive...")
cascade_path = find_file('haar_face.xml', '/content/drive/MyDrive')

if cascade_path is None:
    # Fallback to downloading a verified Haar Cascade file from OpenCV's official repository
    print("Could not find 'haar_face.xml' in your Drive. Downloading official Haar Cascade model...")
    import urllib.request
    cascade_path = '/content/haar_face_downloaded.xml'
    url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    urllib.request.urlretrieve(url, cascade_path)
else:
    print(f"Found cascade file at: {cascade_path}")

# Image path pointing to your Woody photo
img_path = '/content/drive/MyDrive/IT QM/Resources/Photos/Woody.jpg'
img = cv.imread(img_path)

if img is None:
    raise FileNotFoundError(f"Image not found at path: {img_path}")

haar_cascade = cv.CascadeClassifier(cascade_path)

if haar_cascade.empty():
    raise FileNotFoundError(f"Haar cascade model failed to load correctly.")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.equalizeHist(gray)

height, width = gray.shape

faces = []

# ===============================
# 1. MAIN FACE SEARCH AREA
# ===============================
main_x1 = int(width * 0.05)
main_y1 = int(height * 0.30)
main_x2 = int(width * 0.75)
main_y2 = int(height * 0.95)

main_roi = gray[main_y1:main_y2, main_x1:main_x2]

main_faces = haar_cascade.detectMultiScale(
    main_roi,
    scaleFactor=1.05,
    minNeighbors=3,
    minSize=(80, 80),
    maxSize=(350, 350)
)

if len(main_faces) > 0:
    main_face = max(main_faces, key=lambda face: face[2] * face[3])
    x, y, w, h = main_face
    faces.append((x + main_x1, y + main_y1, w, h))

# ===============================
# 2. TIKTOK PROFILE FACE AREA
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

if len(profile_faces) > 0:
    profile_face = max(profile_faces, key=lambda face: face[2] * face[3])
    x, y, w, h = profile_face
    faces.append((x + profile_x1, y + profile_y1, w, h))

# ===============================
# DRAW RESULT
# ===============================
output = img.copy()

for i, (x, y, w, h) in enumerate(faces, start=1):
    cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    cv.putText(
        output,
        f'Face {i}',
        (x, y - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

print(f'Number of real faces found = {len(faces)}')
cv2_imshow(output)
