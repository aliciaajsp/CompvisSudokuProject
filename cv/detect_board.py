import cv2

def find_sudoku_contour(thresh):

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 5000:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.04 * perimeter,
            True
        )

        if len(approx) == 4:
            return approx

    return None