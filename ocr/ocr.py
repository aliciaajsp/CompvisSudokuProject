import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

DEFAULT_MODEL_PATH = MODEL_DIR / "digit_cnn_tmnist.pt"
DEFAULT_CONSTANTS_PATH = MODEL_DIR / "norm_constants_tmnist.npy"
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMBINED_MEAN = None
COMBINED_STD = None


def load_digit_model(model_path=DEFAULT_MODEL_PATH, constants_path=DEFAULT_CONSTANTS_PATH):
    global model, COMBINED_MEAN, COMBINED_STD
    model_path = Path(model_path)
    constants_path = Path(constants_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan di: {model_path}")

    if constants_path.exists():
        constants = np.load(constants_path)
        COMBINED_MEAN = float(constants[0])
        COMBINED_STD = float(constants[1])
    else:
        COMBINED_MEAN = 0.1932
        COMBINED_STD = 0.3241

    model = DigitCNN().to(device)
    state_dict = torch.load(str(model_path), map_location=device)

    if isinstance(state_dict, nn.Module):
        model = state_dict.to(device)
    else:
        model.load_state_dict(state_dict)

    model.eval()
    return model


def enhance_for_blur(img):
    """Placeholder preprocessing — dikembalikan apa adanya, bisa diperluas nanti."""
    return img


def order_pts(pts):
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]],
        dtype="float32"
    )


def count_children(idx, hierarchy):
    count = 0
    child = hierarchy[0][idx][2]
    while child != -1:
        count += 1
        child = hierarchy[0][child][0]
    return count


def is_square_like(contour, tolerance=0.15):
    x, y, w, h = cv2.boundingRect(contour)
    if h == 0:
        return False
    ratio = w / h
    area = cv2.contourArea(contour)
    return (1 - tolerance) < ratio < (1 + tolerance) and area > 10000


def select_grid_contour(contours, hierarchy, img_shape=None):
    square_contours = [c for c in contours if is_square_like(c)]
    if not square_contours:
        return max(contours, key=cv2.contourArea) if contours else None

    img_h, img_w = img_shape if img_shape is not None else (0, 0)
    img_area = img_h * img_w if img_h and img_w else 0
    idx_map = {id(c): i for i, c in enumerate(contours)}

    def score(c):
        idx = idx_map[id(c)]
        n_children = count_children(idx, hierarchy)
        closeness = min(abs(n_children - 81), abs(n_children - 9))
        area = cv2.contourArea(c)
        epsilon = 0.03 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        is_quad = 1 if len(approx) == 4 else 0

        size_penalty = 0
        if img_area > 0:
            area_ratio = area / img_area
            if area_ratio > 0.90:
                size_penalty = 2 
            elif area_ratio > 0.75:
                size_penalty = 1 


        center_bonus = 0
        if img_area > 0:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w / 2, y + h / 2

            in_center = (0.20 * img_w < cx < 0.80 * img_w and
                         0.20 * img_h < cy < 0.80 * img_h)
            center_bonus = -1 if in_center else 0  

        return (-area, size_penalty, center_bonus, -is_quad, closeness)

    square_contours.sort(key=score)
    return square_contours[0]


def remove_cell_border(cell):
    contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cell
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w < cell.shape[1] * 0.15 or h < cell.shape[0] * 0.15:
        return cell
    return cell[y:y + h, x:x + w]


def compute_intersections(warped_img, SIZE):
    lines = cv2.HoughLinesP(
        warped_img, rho=1, theta=np.pi / 180,
        threshold=80, minLineLength=SIZE // 3, maxLineGap=20
    )
    if lines is None:
        return None

    horizontal, vertical = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > abs(y2 - y1):
            horizontal.append((x1, y1, x2, y2))
        else:
            vertical.append((x1, y1, x2, y2))

    def cluster_lines(lines_list, is_horizontal, size_limit):
        lines_list = sorted(
            lines_list,
            key=lambda l: (l[1] + l[3]) / 2 if is_horizontal else (l[0] + l[2]) / 2
        )
        clusters = []
        for line in lines_list:
            coord = (line[1] + line[3]) / 2 if is_horizontal else (line[0] + line[2]) / 2
            if not clusters:
                clusters.append([line])
            else:
                last = clusters[-1][-1]
                last_coord = (last[1] + last[3]) / 2 if is_horizontal else (last[0] + last[2]) / 2
                if abs(coord - last_coord) < (size_limit // 15):
                    clusters[-1].append(line)
                else:
                    clusters.append([line])

        representative_lines = []
        for cluster in clusters:
            xs = [l[0] for l in cluster] + [l[2] for l in cluster]
            ys = [l[1] for l in cluster] + [l[3] for l in cluster]
            if is_horizontal:
                representative_lines.append((min(xs), int(np.mean(ys)), max(xs), int(np.mean(ys))))
            else:
                representative_lines.append((int(np.mean(xs)), min(ys), int(np.mean(xs)), max(ys)))
        return representative_lines

    h_lines = cluster_lines(horizontal, True, SIZE)
    v_lines = cluster_lines(vertical, False, SIZE)

    if len(h_lines) < 10 or len(v_lines) < 10:
        return None

    h_lines = h_lines[:10]
    v_lines = v_lines[:10]

    points_grid = np.zeros((10, 10, 2), dtype="int32")
    for r in range(10):
        for c in range(10):
            points_grid[r][c] = [v_lines[c][0], h_lines[r][1]]

    return points_grid



def validate_sudoku_cell(board, row, col, val):
    if val == 0:
        return True
    for c in range(9):
        if c != col and board[row][c] == val:
            return False
    for r in range(9):
        if r != row and board[r][col] == val:
            return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(start_row, start_row + 3):
        for c in range(start_col, start_col + 3):
            if (r != row or c != col) and board[r][c] == val:
                return False
    return True


def main(image_path="sudoku.png", model_path=DEFAULT_MODEL_PATH, constants_path=DEFAULT_CONSTANTS_PATH):
    global model
    if model is None:
        load_digit_model(model_path, constants_path)

    img_path = Path(image_path)

    if not img_path.is_absolute():
        img_path = BASE_DIR / img_path

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Gambar tidak ditemukan di: {img_path}"

    h, w = img.shape
    SIZE = (min(w, h) // 2)
    SIZE = (SIZE // 9) * 9
    SIZE = max(450, min(SIZE, 1800))
    cell_size = SIZE // 9
    margin = int(cell_size * 0.08)

    img_enhanced = enhance_for_blur(img)
    blur = cv2.GaussianBlur(img_enhanced, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    grid_contour = select_grid_contour(contours, hierarchy, img_shape=img.shape)
    if grid_contour is None:
        raise ValueError("Sudoku grid not detected.")
    epsilon = 0.04 * cv2.arcLength(grid_contour, True)
    approx = cv2.approxPolyDP(grid_contour, epsilon, True)

    if len(approx) == 4:
        src = order_pts(approx)
        dst = np.array([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]], dtype="float32")
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(thresh, M, (SIZE, SIZE))
    else:
        warped = cv2.resize(thresh, (SIZE, SIZE))

    points_grid = compute_intersections(warped, SIZE)

    board = [[0] * 9 for _ in range(9)]
    is_blank = [[False] * 9 for _ in range(9)]
    conf_map = [[0.0] * 9 for _ in range(9)]
    second_best = [[0] * 9 for _ in range(9)]

    for row in range(9):
        for col in range(9):
            if points_grid is not None:
                p1 = points_grid[row][col]
                p2 = points_grid[row][col + 1]
                p3 = points_grid[row + 1][col + 1]
                p4 = points_grid[row + 1][col]
                src_pts = np.array([p1, p2, p3, p4], dtype="float32")
                dst_pts = np.array(
                    [[0, 0], [cell_size, 0], [cell_size, cell_size], [0, cell_size]],
                    dtype="float32"
                )
                M_cell = cv2.getPerspectiveTransform(src_pts, dst_pts)
                cell = cv2.warpPerspective(warped, M_cell, (cell_size, cell_size))
            else:
                y1, y2 = row * cell_size, (row + 1) * cell_size
                x1, x2 = col * cell_size, (col + 1) * cell_size
                cell = warped[y1:y2, x1:x2]

            cell = cell[margin:cell_size - margin, margin:cell_size - margin]

            kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, kernel_clean)

            quick = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_AREA)
            if np.sum(quick > 127) / (28 * 28) < 0.01:
                is_blank[row][col] = True
                continue

            kernel = np.ones((2, 2), np.uint8)
            cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, kernel)
            cell = remove_cell_border(cell)
            cell = cv2.copyMakeBorder(cell, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
            cell_resized = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_AREA)

            inp = cell_resized.astype("float32") / 255.0
            inp = (inp - COMBINED_MEAN) / (COMBINED_STD + 1e-7)
            inp_tensor = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(inp_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]

            sorted_preds = np.argsort(probabilities)[::-1]
            top_digit = int(sorted_preds[0])
            if top_digit == 0:
                top_digit = int(sorted_preds[1])
                rest = sorted_preds[2:]
            else:
                rest = sorted_preds[1:]

            second_digit = int(rest[0])
            if second_digit == 0 and len(rest) > 1:
                second_digit = int(rest[1])

            board[row][col] = top_digit
            conf_map[row][col] = float(np.max(probabilities))
            second_best[row][col] = second_digit

    raw_board = [r[:] for r in board]
    cells_by_conf = sorted(
        [(r, c) for r in range(9) for c in range(9) if not is_blank[r][c]],
        key=lambda rc: conf_map[rc[0]][rc[1]],
        reverse=True
    )
    board = [[0] * 9 for _ in range(9)]
    for r, c in cells_by_conf:
        val = raw_board[r][c]
        if validate_sudoku_cell(board, r, c, val):
            board[r][c] = val
        else:
            alt = second_best[r][c]
            if alt != 0 and validate_sudoku_cell(board, r, c, alt):
                board[r][c] = alt
            else:
                board[r][c] = 0
                is_blank[r][c] = True

    return board


if __name__ == "__main__":
    result = main(image_path="sudoku.png")
