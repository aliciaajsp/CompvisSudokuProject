import cv2
import os

def split_cells(board, save=False, output_dir="output/cells"):
    rows = []
    height, width = board.shape[:2]

    cell_height = height // 9
    cell_width = width // 9

    if save:
        os.makedirs(output_dir, exist_ok=True)
    
    for row in range(9):
        cols = []
        for col in range(9):
            x1 = col * cell_width
            y1 = row * cell_height

            x2 = (col + 1) * cell_width
            y2 = (row + 1) * cell_height

            cell = board[y1:y2, x1:x2]
            cols.append(cell)

            if save:
                filename = f"cell_{row}_{col}.jpg"
                path = os.path.join(output_dir, filename)
                cv2.imwrite(path, cell)
        
        rows.append(cols)
    return rows