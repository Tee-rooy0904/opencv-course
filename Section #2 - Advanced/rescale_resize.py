#pylint:disable=no-member

import cv2 as cv

# img = cv.imread('../Resources/Photos/cat.jpg')
# cv.imshow('Cat', img)
#pylint:disable=no-member

import cv2 as cv
from google.colab.patches import cv2_imshow

def rescaleFrame(frame, scale=0.75):
    # Images, Videos and Live Video
    if frame is None:
        return None
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Reading Videos using absolute path from Google Drive
video_path = '/content/drive/MyDrive/IT QM/Resources/Videos/Haha.mp4'
capture = cv.VideoCapture(video_path)

if not capture.isOpened():
    print(f"Error: Could not open video file from '{video_path}'. Please verify the file path.")
else:
    frame_count = 0
    max_frames_to_display = 5  # Limits rendering to prevent overflowing the notebook output

    while True:
        isTrue, frame = capture.read()
        
        if not isTrue or frame is None:
            break

        frame_resized = rescaleFrame(frame, scale=.2)
        
        print(f"--- Displaying Frame {frame_count + 1} ---")
        print("Original Frame:")
        cv2_imshow(frame)
        print("Resized Frame:")
        cv2_imshow(frame_resized)

        frame_count += 1
        if frame_count >= max_frames_to_display:
            print(f"Displayed {max_frames_to_display} frames of the video for visualization.")
            break

capture.release()
def rescaleFrame(frame, scale=0.75):
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width,height):
    # Live video
    capture.set(3,width)
    capture.set(4,height)
    
# Reading Videos
capture = cv.VideoCapture('../Resources/Videos/Haha.mp4')

while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame, scale=.2)
    
    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()
