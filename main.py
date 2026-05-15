import cv2

from cv.preprocessing import preprocess_image
from cv.detect_board import find_sudoku_contour

image, thresh = preprocess_image(
    "test_images/sample12.png"
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

# testing dcaa
# perspective_transform
from segmentation.perspective_transform import (warp_perspective)

warped = warp_perspective(image, corners)
cv2.imshow("Original", image)
cv2.imshow("Warped Version", warped)
cv2.waitKey(0)

# split_cells
from segmentation.split_cells import (split_cells)

cells = split_cells(warped, save=True)
cv2.imshow("Cell 0 0", cells[0][0]) # ganti disini mau cell keberapa (1-8)
cv2.waitKey(0)

# crop_cells
from segmentation.crop_cells import (clean_cell)

os.makedirs("outputs/cleaned_cells", exist_ok=True)
cell = cv2.imread("outputs/cells/cell_0_1.jpg")
cell = cells[0][2] # ganti disini buat cek cell lain
cleaned = clean_cell(cell)
cv2.imwrite("output/cleaned_cells/cleaned_0_1.jpg", cleaned)
cv2.imshow("Original Cell", cell)
cv2.imshow("Cleaned Cell", cleaned)
cv2.waitKey(0)

# utils
from segmentation.utils import (show_image, save_image, draw_grid, stack_images)

grid_image = draw_grid(warped)
save_image("outputs/grid/grid.jpg", grid_image)
combined = stack_images([warped, grid_image])
show_image("Comparison", combined)
