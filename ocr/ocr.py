import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn

# model definition
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

# config
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
DEFAULT_MODEL_PATH = MODEL_DIR / "digit_cnn_tmnist.pt"
DEFAULT_CONSTANTS_PATH = MODEL_DIR / "norm_constants_tmnist.npy"   
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMBINED_MEAN = None
COMBINED_STD = None

# load model weights & normalization constants
def load_digit_model(model_path=DEFAULT_MODEL_PATH, constants_path=DEFAULT_CONSTANTS_PATH):
    global model, COMBINED_MEAN, COMBINED_STD
    model_path = Path(model_path)
    constants_path = Path(constants_path)

    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model tidak ditemukan di: {model_path}")

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

# image utilities
def enhance_for_blur(img):
    gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)
    return cv2.fastNlMeansDenoising(enhanced, h=7, templateWindowSize=7, searchWindowSize=21)

def order_pts(pts):
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]], dtype="float32")

# sudoku rule validation
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

# ocr pipeline
def main(image_path="sudoku.png", model_path=DEFAULT_MODEL_PATH, constants_path=DEFAULT_CONSTANTS_PATH):
    global model
    if model is None:
        load_digit_model(model_path, constants_path)

    img_path = Path(image_path)
    if not img_path.is_absolute():
        img_path = BASE_DIR / img_path

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"❌ Gambar tidak ditemukan di: {img_path}"

    h, w = img.shape
    SIZE = (min(w, h) // 2)
    SIZE = (SIZE // 9) * 9
    SIZE = max(450, min(SIZE, 1800))
    cell_size = SIZE // 9
    margin = int(cell_size * 0.08)

    img_enhanced = enhance_for_blur(img)
    blur = cv2.GaussianBlur(img_enhanced, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 6)

    # grid detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    def is_square_like(contour, tolerance=0.3):
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0: return False
        ratio = w / h
        area = cv2.contourArea(contour)
        return (1 - tolerance) < ratio < (1 + tolerance) and area > 10000

    square_contours = [c for c in contours if is_square_like(c)]
    grid_contour = max(square_contours, key=cv2.contourArea) if square_contours else max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(grid_contour, True)
    approx = cv2.approxPolyDP(grid_contour, epsilon, True)

    if len(approx) == 4:
        src = order_pts(approx)
        dst = np.array([[0,0],[SIZE,0],[SIZE,SIZE],[0,SIZE]], dtype="float32")
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(thresh, M, (SIZE, SIZE))
    else:
        warped = cv2.resize(thresh, (SIZE, SIZE))

    board = [[0]*9 for _ in range(9)]
    is_blank = [[False]*9 for _ in range(9)]
    conf_map = [[0.0]*9 for _ in range(9)]
    second_best = [[0]*9 for _ in range(9)]

    # digit recognition
    for row in range(9):
        for col in range(9):
            y1, y2 = row * cell_size, (row + 1) * cell_size
            x1, x2 = col * cell_size, (col + 1) * cell_size
            cell = warped[y1:y2, x1:x2]
            cell = cell[margin:cell_size-margin, margin:cell_size-margin]

            quick = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_AREA)
            pixel_density = np.sum(quick > 127) / (28 * 28)
            if pixel_density < 0.075:
                is_blank[row][col] = True
                continue

            kernel = np.ones((2, 2), np.uint8)
            cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, kernel)
            cell_resized = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_AREA)

            inp = cell_resized.astype("float32") / 255.0
            inp = (inp - COMBINED_MEAN) / (COMBINED_STD + 1e-7)
            inp_tensor = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(inp_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]

            sorted_predictions = np.argsort(probabilities)[::-1]
            top_digit = int(sorted_predictions[0])
            if top_digit == 0:
                top_digit = int(sorted_predictions[1])
                subsequent_predictions = sorted_predictions[2:]
            else:
                subsequent_predictions = sorted_predictions[1:]

            second_digit = int(subsequent_predictions[0])
            if second_digit == 0 and len(subsequent_predictions) > 1:
                second_digit = int(subsequent_predictions[1])

            board[row][col] = top_digit
            conf_map[row][col] = float(np.max(probabilities))
            second_best[row][col] = second_digit

    # correction & validation
    raw_board = [row[:] for row in board]
    cells_by_conf = sorted(
        [(row, col) for row in range(9) for col in range(9) if not is_blank[row][col]],
        key=lambda rc: conf_map[rc[0]][rc[1]],
        reverse=True
    )
    board = [[0]*9 for _ in range(9)]
    for row, col in cells_by_conf:
        current_val = raw_board[row][col]

        if validate_sudoku_cell(board, row, col, current_val):
            board[row][col] = current_val
        else:
            alt_val = second_best[row][col]
            if validate_sudoku_cell(board, row, col, alt_val) and alt_val != 0:
                board[row][col] = alt_val
            else:
                board[row][col] = 0
                is_blank[row][col] = True
    # matrix output
    for i in range(9):
        comma = "," if i < 8 else ""
        print(f"  {[0 if is_blank[i][c] else board[i][c] for c in range(9)]}{comma}")
    return board

if __name__ == "__main__":
    main(image_path="sudoku7.png")
