import cv2 as cv
from google.colab.patches import cv2_imshow

# Read in an image
img = cv.imread('/content/drive/MyDrive/IT QM/Resources/Photos/Woody.jpg')

# Check if image was loaded successfully
if img is None:
    print("Error: Could not load image from '/content/drive/MyDrive/IT QM/Resources/Photos/Woody.jpg'. Please check the path.")
else:
    cv2_imshow(img)

# Reading Videos
capture = cv.VideoCapture('/content/drive/MyDrive/IT QM/Resources/Videos/Haha.mp4')

if not capture.isOpened():
    print("Error: Could not open video file.")
else:
    frame_count = 0
    max_frames_to_display = 10  # Display a limited number of frames for demonstration in Colab

    while True:
        isTrue, frame = capture.read()

        if isTrue:
            cv2_imshow(frame)
            frame_count += 1
            # Break after displaying a few frames
            if frame_count >= max_frames_to_display:
                print(f"Displayed {max_frames_to_display} frames of the video.")
                break
        else:
            break

capture.release()
# cv.destroyAllWindows() is not needed for cv2_imshow
