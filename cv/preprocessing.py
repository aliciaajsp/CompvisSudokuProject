import cv2

def preprocess_image(path):

    image = cv2.imread(path)

    image = cv2.resize(image, (600, 600))

    image = cv2.copyMakeBorder(
    image,
    20, 20, 20, 20,
    cv2.BORDER_CONSTANT,
    value=[255, 255, 255]
)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

    return image, thresh, gray, blur