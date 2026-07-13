# pylint:disable=no-member

import os
import cv2 as cv
import numpy as np


# Folder names must exactly match the folders inside Resources/Faces/train
PEOPLE = [
    'Ben Affleck',
    'Elton John',
    'Jerry Seinfeld',
    'Madonna',
    'Mindy Kaling'
]

# Set to True to display every training image.
SHOW_PREVIEW = True

# How long each image stays visible in milliseconds.
PREVIEW_DELAY = 500


# Get the folder where faces_train.py is located.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# faces_train.py is inside "Section #3 - Faces".
# Resources is one folder above it.
TRAIN_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR,
        '..',
        'Resources',
        'Faces',
        'train'
    )
)

# haar_face.xml is inside the same folder as faces_train.py.
HAAR_PATH = os.path.join(BASE_DIR, 'haar_face.xml')

haar_cascade = cv.CascadeClassifier(HAAR_PATH)

features = []
labels = []


def resize_for_preview(image, maximum_width=800, maximum_height=600):
    """Resize large images so they fit properly on the screen."""

    height, width = image.shape[:2]

    width_scale = maximum_width / width
    height_scale = maximum_height / height

    scale = min(width_scale, height_scale, 1.0)

    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)

        return cv.resize(
            image,
            (new_width, new_height),
            interpolation=cv.INTER_AREA
        )

    return image


def create_train():
    """Read the images, detect faces, and collect training data."""

    global SHOW_PREVIEW

    print(f'Training directory:\n{TRAIN_DIR}\n')

    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            f'Training directory was not found:\n{TRAIN_DIR}'
        )

    if haar_cascade.empty():
        raise FileNotFoundError(
            f'haar_face.xml could not be loaded from:\n{HAAR_PATH}'
        )

    for label, person in enumerate(PEOPLE):
        person_folder = os.path.join(TRAIN_DIR, person)

        if not os.path.isdir(person_folder):
            print(f'Folder missing, skipped: {person_folder}')
            continue

        print(f'Processing: {person}')

        image_files = os.listdir(person_folder)

        for image_name in image_files:
            image_path = os.path.join(person_folder, image_name)

            # Ignore folders or non-file entries.
            if not os.path.isfile(image_path):
                continue

            image_array = cv.imread(image_path)

            if image_array is None:
                print(f'  Unreadable image skipped: {image_name}')
                continue

            gray = cv.cvtColor(
                image_array,
                cv.COLOR_BGR2GRAY
            )

            faces_rect = haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4
            )

            preview = image_array.copy()

            if len(faces_rect) == 0:
                print(f'  No face detected: {image_name}')

            for x, y, width, height in faces_rect:
                face_roi = gray[
                    y:y + height,
                    x:x + width
                ]

                features.append(face_roi)
                labels.append(label)

                # Draw a rectangle around each detected face.
                cv.rectangle(
                    preview,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    thickness=2
                )

                cv.putText(
                    preview,
                    person,
                    (x, max(y - 10, 20)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    thickness=2
                )

            if SHOW_PREVIEW:
                preview = resize_for_preview(preview)

                cv.imshow(
                    'Face Training Preview - Press Q to stop preview',
                    preview
                )

                key = cv.waitKey(PREVIEW_DELAY) & 0xFF

                # Pressing Q only closes the preview.
                # Training will continue in the background.
                if key == ord('q'):
                    SHOW_PREVIEW = False
                    cv.destroyAllWindows()

        print(f'Finished: {person}\n')


def train_and_save():
    """Train the LBPH recognizer and save the generated files."""

    create_train()
    cv.destroyAllWindows()

    if len(features) == 0:
        raise RuntimeError(
            'No faces were detected from the training images. '
            'Check the images and haar_face.xml.'
        )

    labels_array = np.array(labels, dtype=np.int32)
    features_array = np.array(features, dtype=object)

    if not hasattr(cv, 'face'):
        raise ImportError(
            'cv.face is unavailable. Install OpenCV Contrib using:\n'
            'pip install opencv-contrib-python'
        )

    face_recognizer = cv.face.LBPHFaceRecognizer_create()

    # Train using the detected face images and their labels.
    face_recognizer.train(
        features,
        labels_array
    )

    trained_model_path = os.path.join(
        BASE_DIR,
        'face_trained.yml'
    )

    features_path = os.path.join(
        BASE_DIR,
        'features.npy'
    )

    labels_path = os.path.join(
        BASE_DIR,
        'labels.npy'
    )

    face_recognizer.save(trained_model_path)

    np.save(
        features_path,
        features_array,
        allow_pickle=True
    )

    np.save(
        labels_path,
        labels_array
    )

    print('Training completed successfully.')
    print(f'Detected faces: {len(features)}')
    print('\nFiles saved:')
    print(trained_model_path)
    print(features_path)
    print(labels_path)


if __name__ == '__main__':
    try:
        train_and_save()

    except Exception as error:
        cv.destroyAllWindows()
        print('\nAn error occurred:')
        print(error)