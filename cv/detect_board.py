import cv2


def find_sudoku_contour(thresh):

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    largest_contour = None
    max_area = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > 1000 and area > max_area:

            peri = cv2.arcLength(contour, True)

            approx = cv2.approxPolyDP(
                contour,
                0.04 * peri,
                True
            )

            if len(approx) == 4:
                largest_contour = approx
                max_area = area

    return largest_contour