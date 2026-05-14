import cv2
import os
import numpy as np

def show_image(title, image, width=600, height=600):
    resized = cv2.resize(image, (width, height))
    cv2.imshow(title, resized)
    cv2.waitKey(0)

def save_image(path, image):
    folder = os.path.dirname(path)

    if folder != "":
        os.makedirs(folder, exist_ok=True)
    
    cv2.imwrite(path, image)

def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

def draw_grid(image, rows=9, cols=9):
    output = image.copy()
    height, width = output.shape[:2]
    cell_height = height // rows
    cell_width = width // cols

    # garis hrorizontal
    for i in range (rows + 1):
        y = i * cell_height
        cv2.line(output, (0, y), (width, y), (0, 255, 0), 1)
    
    # garis vertikal
    for j in range (cols + 1):
        x = j * cell_width
        cv2.line(output, (x, 0), (x, height), (0, 255, 0), 1)
    
    return output

def stack_images(images, scale = 1):
    resized_images = []

    for image in images:
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        resized = cv2.resize(image, (width, height))

        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        
        resized_images.append(resized)
    
    return np.hstack(resized_images)