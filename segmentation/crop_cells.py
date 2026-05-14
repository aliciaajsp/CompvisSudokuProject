import cv2
import numpy as np

def crop_cell(cell, margin=8):
    height, width = cell.shape[:2]

    cropped = cell[
        margin : height - margin,
        margin : width - margin
    ]
    return cropped

def preprocess_cell(cell):
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    return thresh

def remove_small_noise(cell, min_area = 50):
    contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(cell)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask

def clean_cell(cell):
    cropped = crop_cell(cell)
    processed = preprocess_cell(cropped)
    cleaned = remove_small_noise(processed)
    return cleaned