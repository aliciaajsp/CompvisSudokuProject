import cv2

from cv.preprocessing import preprocess_image
from cv.detect_board import find_sudoku_contour

image, thresh, gray, blur = preprocess_image(
    "test_images/sample3.png"
)

corners = find_sudoku_contour(thresh)

if corners is not None:

    cv2.drawContours(
        image,
        [corners],
        -1,
        (0, 255, 0),
        5
    )

    for point in corners:
        x, y = point[0]

        cv2.circle(
            image,
            (x, y),
            10,
            (0, 0, 255),
            -1
        )

    print("Corner Points:")
    print(corners)

else:
    print("Sudoku board not detected.")

cv2.imshow("Sudoku Detection", image)

cv2.waitKey(0)
cv2.destroyAllWindows()