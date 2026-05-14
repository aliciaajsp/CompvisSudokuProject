import cv2
import numpy as np

def reorder_points(points):
    # urutan corner : kiri atas, kanan atas, kanan bawah, kiri bawah
    points = points.reshape((4, 2))
    ordered = np.zeros((4, 2), dtype=np.float32)

    # total semua koordinat
    total = points.sum(axis=1)
    ordered[0] = points[np.argmin(total)] # kiri atas
    ordered[2] = points[np.argmax(total)] # kanan bawah

    # selisih koordinat
    diff = np.diff(points, axis=1)
    ordered[1] = points[np.argmin(diff)] # kanan atas
    ordered[3] = points[np.argmax(diff)] # kiri bawah

    return ordered

def warp_perspective(image, corners, size=450):
    ordered_corners = reorder_points(corners)

    destination = np.array([[0, 0], [size - 1, 0], [size - 1, size- 1], [0, size - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(ordered_corners, destination)
    warped = cv2.warpPerspective(image, matrix, (size, size))

    return warped