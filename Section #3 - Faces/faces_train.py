# pylint:disable=no-member

import os
import cv2 as cv
import numpy as np


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

SHOW_PREVIEW = True
PREVIEW_DELAY = 500

# All saved face samples will use this size.
FACE_SIZE = (200, 200)

# Cartoon folders that require an automatic special fallback.
CARTOON_PEOPLE = {
    'Woody'
}


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TRAIN_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR,
        '..',
        'Resources',
        'Faces',
        'train'
    )
)

HAAR_PATH = os.path.join(
    BASE_DIR,
    'haar_face.xml'
)

haar_cascade = cv.CascadeClassifier(
    HAAR_PATH
)

features = []
labels = []


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def resize_for_preview(
    image,
    max_width=900,
    max_height=650
):
    """Resize large images so they fit on the screen."""

    height, width = image.shape[:2]

    scale_width = max_width / width
    scale_height = max_height / height

    scale = min(
        scale_width,
        scale_height,
        1.0
    )

    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)

        return cv.resize(
            image,
            (new_width, new_height),
            interpolation=cv.INTER_AREA
        )

    return image.copy()


def get_people():
    """Automatically read all folders inside Resources/Faces/train."""

    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            f'Training directory was not found:\n'
            f'{TRAIN_DIR}'
        )

    people = []

    for folder_name in os.listdir(TRAIN_DIR):
        folder_path = os.path.join(
            TRAIN_DIR,
            folder_name
        )

        if os.path.isdir(folder_path):
            people.append(folder_name)

    people.sort()

    if not people:
        raise RuntimeError(
            f'No person folders were found inside:\n'
            f'{TRAIN_DIR}'
        )

    return people


def clamp_rectangle(
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height
):
    """Keep a rectangle inside the image boundaries."""

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))

    x2 = min(image_width, int(x2))
    y2 = min(image_height, int(y2))

    return x1, y1, x2, y2


def detect_human_face(gray_image):
    """
    Detect the largest valid human face.

    Small eye-sized detections are rejected.
    """

    image_height, image_width = gray_image.shape

    smallest_dimension = min(
        image_width,
        image_height
    )

    minimum_size = max(
        60,
        int(smallest_dimension * 0.15)
    )

    faces = haar_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.08,
        minNeighbors=6,
        minSize=(
            minimum_size,
            minimum_size
        )
    )

    image_area = image_width * image_height
    valid_faces = []

    for x, y, width, height in faces:
        face_area = width * height
        aspect_ratio = width / float(height)

        large_enough = (
            face_area >= image_area * 0.035
        )

        reasonable_shape = (
            0.60 <= aspect_ratio <= 1.50
        )

        if large_enough and reasonable_shape:
            valid_faces.append(
                (x, y, width, height)
            )

    if not valid_faces:
        return None

    return max(
        valid_faces,
        key=lambda rectangle:
        rectangle[2] * rectangle[3]
    )


def expand_human_face(
    face_rectangle,
    image_width,
    image_height
):
    """Expand a detected human face slightly."""

    x, y, width, height = face_rectangle

    horizontal_padding = int(width * 0.18)
    top_padding = int(height * 0.25)
    bottom_padding = int(height * 0.15)

    return clamp_rectangle(
        x - horizontal_padding,
        y - top_padding,
        x + width + horizontal_padding,
        y + height + bottom_padding,
        image_width,
        image_height
    )


def automatic_cartoon_face(
    gray_image
):
    """
    Automatically obtain a larger face region for a cartoon.

    Haar Cascade may detect an eye or a small facial section.
    This function expands the largest detected area into a full-face crop.
    If nothing is detected, it uses a centered crop automatically.
    """

    image_height, image_width = gray_image.shape

    # Use less strict settings only to find a possible facial landmark.
    candidates = haar_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.05,
        minNeighbors=2,
        minSize=(20, 20)
    )

    smallest_dimension = min(
        image_width,
        image_height
    )

    # The final cartoon crop will be much larger than an eye.
    crop_size = int(
        smallest_dimension * 0.65
    )

    if len(candidates) > 0:
        # Select the largest detected section.
        x, y, width, height = max(
            candidates,
            key=lambda rectangle:
            rectangle[2] * rectangle[3]
        )

        center_x = x + width // 2
        center_y = y + height // 2

        # Move the crop slightly downward if the detected area is an eye.
        center_y += int(crop_size * 0.10)

    else:
        # Fully automatic centered fallback.
        center_x = image_width // 2
        center_y = int(image_height * 0.43)

    half_size = crop_size // 2

    x1 = center_x - half_size
    y1 = center_y - half_size
    x2 = center_x + half_size
    y2 = center_y + half_size

    return clamp_rectangle(
        x1,
        y1,
        x2,
        y2,
        image_width,
        image_height
    )


def show_preview(
    image,
    person,
    rectangle,
    method
):
    """Display the selected training area."""

    preview = image.copy()

    x1, y1, x2, y2 = rectangle

    cv.rectangle(
        preview,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        thickness=2
    )

    cv.putText(
        preview,
        f'{person} - {method}',
        (x1, max(y1 - 10, 25)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        thickness=2
    )

    preview = resize_for_preview(
        preview
    )

    cv.imshow(
        'Automatic Face Training - Press Q to close preview',
        preview
    )

    return cv.waitKey(
        PREVIEW_DELAY
    ) & 0xFF


# ---------------------------------------------------------
# TRAINING IMAGE SCANNING
# ---------------------------------------------------------

def create_train(people):
    """Scan every folder and automatically collect face samples."""

    global SHOW_PREVIEW

    if haar_cascade.empty():
        raise FileNotFoundError(
            f'haar_face.xml could not be loaded from:\n'
            f'{HAAR_PATH}'
        )

    valid_extensions = (
        '.jpg',
        '.jpeg',
        '.png',
        '.bmp',
        '.webp'
    )

    total_images = 0
    added_images = 0
    skipped_images = 0

    print('\nTraining directory:')
    print(TRAIN_DIR)

    print('\nFolders found:')

    for person in people:
        print(f'  - {person}')

    print()

    for label, person in enumerate(people):
        person_folder = os.path.join(
            TRAIN_DIR,
            person
        )

        print(f'Processing: {person}')

        image_names = sorted(
            os.listdir(person_folder)
        )

        for image_name in image_names:
            if not image_name.lower().endswith(
                valid_extensions
            ):
                continue

            total_images += 1

            image_path = os.path.join(
                person_folder,
                image_name
            )

            image_array = cv.imread(
                image_path
            )

            if image_array is None:
                print(
                    f'  Unreadable image skipped: '
                    f'{image_name}'
                )

                skipped_images += 1
                continue

            gray = cv.cvtColor(
                image_array,
                cv.COLOR_BGR2GRAY
            )

            gray_equalized = cv.equalizeHist(
                gray
            )

            image_height, image_width = gray.shape

            if person in CARTOON_PEOPLE:
                selected_rectangle = (
                    automatic_cartoon_face(
                        gray_equalized
                    )
                )

                detection_method = (
                    'automatic cartoon crop'
                )

            else:
                detected_face = detect_human_face(
                    gray_equalized
                )

                if detected_face is None:
                    print(
                        f'  No complete face detected: '
                        f'{image_name}'
                    )

                    skipped_images += 1
                    continue

                selected_rectangle = expand_human_face(
                    detected_face,
                    image_width,
                    image_height
                )

                detection_method = (
                    'automatic detection'
                )

            x1, y1, x2, y2 = selected_rectangle

            face_roi = gray[
                y1:y2,
                x1:x2
            ]

            if face_roi.size == 0:
                print(
                    f'  Invalid crop skipped: '
                    f'{image_name}'
                )

                skipped_images += 1
                continue

            face_roi = cv.resize(
                face_roi,
                FACE_SIZE,
                interpolation=cv.INTER_AREA
            )

            features.append(
                face_roi
            )

            labels.append(
                label
            )

            added_images += 1

            print(
                f'  Face added: {image_name} '
                f'({detection_method})'
            )

            if SHOW_PREVIEW:
                key = show_preview(
                    image_array,
                    person,
                    selected_rectangle,
                    detection_method
                )

                if key == ord('q'):
                    SHOW_PREVIEW = False
                    cv.destroyAllWindows()

        print(f'Finished: {person}\n')

    cv.destroyAllWindows()

    print('Scanning summary:')
    print(f'  Person folders: {len(people)}')
    print(f'  Images checked: {total_images}')
    print(f'  Images added: {added_images}')
    print(f'  Images skipped: {skipped_images}')
    print(f'  Face samples: {len(features)}')


# ---------------------------------------------------------
# TRAINING AND SAVING
# ---------------------------------------------------------

def train_and_save():
    """Train the LBPH recognizer and save its files."""

    people = get_people()

    create_train(
        people
    )

    if not features:
        raise RuntimeError(
            'No faces were collected. '
            'The recognizer cannot be trained.'
        )

    if not hasattr(cv, 'face'):
        raise ImportError(
            'cv.face is unavailable.\n'
            'Install OpenCV Contrib using:\n'
            'pip install opencv-contrib-python'
        )

    labels_array = np.array(
        labels,
        dtype=np.int32
    )

    features_array = np.array(
        features,
        dtype=object
    )

    people_array = np.array(
        people,
        dtype=str
    )

    face_recognizer = (
        cv.face.LBPHFaceRecognizer_create()
    )

    face_recognizer.train(
        features,
        labels_array
    )

    model_path = os.path.join(
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

    people_path = os.path.join(
        BASE_DIR,
        'people.npy'
    )

    face_recognizer.save(
        model_path
    )

    np.save(
        features_path,
        features_array,
        allow_pickle=True
    )

    np.save(
        labels_path,
        labels_array
    )

    np.save(
        people_path,
        people_array
    )

    print('\nTraining completed successfully.')
    print(f'Total face samples: {len(features)}')

    print('\nFiles saved:')
    print(model_path)
    print(features_path)
    print(labels_path)
    print(people_path)


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == '__main__':
    try:
        train_and_save()

    except Exception as error:
        cv.destroyAllWindows()

        print('\nAn error occurred:')
        print(error)