import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch

# config

CONFIG = {
    "test_dir": "test_images",
    "gt_dir": "ground_truth",
    "model_path": "model/digit_cnn_tmnist.pt",
    "constants_path": "model/norm_constants_tmnist.npy",
    "iou_threshold": 0.9,
    "robustness_n": 10,
    "seed": 42,
    "output_file": "report.md",
    "blank_threshold": 0.03
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib
import ocr.ocr as ocr_module
importlib.reload(ocr_module)


# Core pipeline

def process_array(img, ocr_module, blank_threshold=None):
    if blank_threshold is None:
        blank_threshold = CONFIG["blank_threshold"]

    h, w = img.shape
    SIZE = (min(w, h) // 2)
    SIZE = (SIZE // 9) * 9
    SIZE = max(450, min(SIZE, 1800))
    cell_size = SIZE // 9
    margin = int(cell_size * 0.08)

    img_enhanced = ocr_module.enhance_for_blur(img)
    blur = cv2.GaussianBlur(img_enhanced, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    grid_contour = ocr_module.select_grid_contour(contours, hierarchy, img_shape=img.shape)

    if grid_contour is None:
        raise ValueError("Tidak dapat mendeteksi grid Sudoku.")

    epsilon = 0.04 * cv2.arcLength(grid_contour, True)
    approx = cv2.approxPolyDP(grid_contour, epsilon, True)

    used_perspective = len(approx) == 4
    corners = None
    if used_perspective:
        corners = ocr_module.order_pts(approx)
        dst = np.array([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]], dtype="float32")
        M = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(thresh, M, (SIZE, SIZE))
    else:
        warped = cv2.resize(thresh, (SIZE, SIZE))

    points_grid = ocr_module.compute_intersections(warped, SIZE)

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
            if np.sum(quick > 127) / (28 * 28) < blank_threshold:
                is_blank[row][col] = True
                continue

            kernel = np.ones((2, 2), np.uint8)
            cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, kernel)
            cell = ocr_module.remove_cell_border(cell)
            cell = cv2.copyMakeBorder(cell, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
            cell_resized = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_AREA)

            inp = cell_resized.astype("float32") / 255.0
            inp = (inp - ocr_module.COMBINED_MEAN) / (ocr_module.COMBINED_STD + 1e-7)
            inp_t = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(ocr_module.device)

            with torch.no_grad():
                out = ocr_module.model(inp_t)
                probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]

            order = np.argsort(probs)[::-1]
            top = int(order[0])
            if top == 0:
                top = int(order[1])
                rest = order[2:]
            else:
                rest = order[1:]
            second = int(rest[0])
            if second == 0 and len(rest) > 1:
                second = int(rest[1])

            board[row][col] = top
            conf_map[row][col] = float(np.max(probs))
            second_best[row][col] = second

    raw_board = [r[:] for r in board]
    cells_by_conf = sorted(
        [(r, c) for r in range(9) for c in range(9) if not is_blank[r][c]],
        key=lambda rc: conf_map[rc[0]][rc[1]],
        reverse=True
    )
    board = [[0] * 9 for _ in range(9)]
    for r, c in cells_by_conf:
        val = raw_board[r][c]
        if ocr_module.validate_sudoku_cell(board, r, c, val):
            board[r][c] = val
        else:
            alt = second_best[r][c]
            if alt != 0 and ocr_module.validate_sudoku_cell(board, r, c, alt):
                board[r][c] = alt
            else:
                board[r][c] = 0

    return {
        "board": board,
        "raw_board": raw_board,
        "used_perspective": used_perspective,
        "corners": corners,
        "img_shape": img.shape,
    }


def run_pipeline(img_path, ocr_module):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan di: {img_path}")
    t0 = time.perf_counter()
    out = process_array(img, ocr_module)
    out["elapsed"] = time.perf_counter() - t0
    return out



# Metric 1 n 2 Helpers

def polygon_iou(poly_a, poly_b, shape):
    mask_a = np.zeros(shape, dtype=np.uint8)
    mask_b = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask_a, [poly_a.astype(np.int32)], 1)
    cv2.fillPoly(mask_b, [poly_b.astype(np.int32)], 1)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union else 0.0


def cell_polygons_from_quad(quad, n=9):
    TL, TR, BR, BL = quad
    pts = np.zeros((n + 1, n + 1, 2), dtype=np.float32)
    for i in range(n + 1):
        v = i / n
        left = TL + v * (BL - TL)
        right = TR + v * (BR - TR)
        for j in range(n + 1):
            u = j / n
            pts[i, j] = left + u * (right - left)
    cells = {}
    for r in range(n):
        for c in range(n):
            cells[(r, c)] = np.array(
                [pts[r, c], pts[r, c + 1], pts[r + 1, c + 1], pts[r + 1, c]],
                dtype=np.float32,
            )
    return cells


# Metric 4 Helper

def solve_sudoku(board):
    b = [row[:] for row in board]

    def find_empty():
        for r in range(9):
            for c in range(9):
                if b[r][c] == 0:
                    return r, c
        return None

    def valid(r, c, v):
        for cc in range(9):
            if cc != c and b[r][cc] == v:
                return False
        for rr in range(9):
            if rr != r and b[rr][c] == v:
                return False
        sr, sc = 3 * (r // 3), 3 * (c // 3)
        for rr in range(sr, sr + 3):
            for cc in range(sc, sc + 3):
                if (rr, cc) != (r, c) and b[rr][cc] == v:
                    return False
        return True

    def backtrack():
        pos = find_empty()
        if pos is None:
            return True
        r, c = pos
        for v in range(1, 10):
            if valid(r, c, v):
                b[r][c] = v
                if backtrack():
                    return True
                b[r][c] = 0
        return False

    return b if backtrack() else None



# Metric 3 / 6 Helper

def clue_accuracy(pred_board, gt_board):
    correct, total = 0, 0
    for r in range(9):
        for c in range(9):
            if gt_board[r][c] != 0:
                total += 1
                if pred_board[r][c] == gt_board[r][c]:
                    correct += 1
    return 100.0 * correct / total if total else None



# Metric 6 Helpers

def low_light(img, gamma=1.5, brightness_scale=0.6, noise_std=3):
    f = img.astype(np.float32) / 255.0
    f = f * brightness_scale
    f = np.power(np.clip(f, 0, 1), gamma)
    out = np.clip(f * 255.0, 0, 255)
    noise = np.random.normal(0, noise_std, img.shape)
    return np.clip(out + noise, 0, 255).astype(np.uint8)


def extreme_angle(img, max_tilt_deg=10, max_skew_frac=0.05):
    h, w = img.shape
    angle = np.random.uniform(-max_tilt_deg, max_tilt_deg)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=255)
    dx, dy = w * max_skew_frac, h * max_skew_frac
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [np.random.uniform(0, dx), np.random.uniform(0, dy)],
        [w - np.random.uniform(0, dx), np.random.uniform(0, dy)],
        [w - np.random.uniform(0, dx), h - np.random.uniform(0, dy)],
        [np.random.uniform(0, dx), h - np.random.uniform(0, dy)],
    ])
    M2 = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(rotated, M2, (w, h), borderValue=255)



# Threshold Sweep

def sweep_threshold(results, thresholds=None):
    if thresholds is None:
        thresholds = [0.002, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15]

    print(f"{'Threshold':>10} | {'FP':>4} | {'FN':>4} | {'Digit Acc%':>10}")
    print("-" * 45)

    for t in thresholds:
        fp, fn, correct, total_clue = 0, 0, 0, 0
        for img_path, gt, _ in results:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            try:
                out = process_array(img, ocr_module, blank_threshold=t)
                for r in range(9):
                    for c in range(9):
                        gt_val = gt["board"][r][c]
                        pred_val = out["board"][r][c]
                        if gt_val == 0 and pred_val != 0:
                            fp += 1
                        elif gt_val != 0 and pred_val == 0:
                            fn += 1
                        elif gt_val != 0:
                            total_clue += 1
                            if pred_val == gt_val:
                                correct += 1
            except Exception as e:
                pass
        digit_acc = 100 * correct / total_clue if total_clue else 0
        print(f"{t:>10.3f} | {fp:>4} | {fn:>4} | {digit_acc:>9.1f}%")



# Main Evaluation

def main_eval():
    np.random.seed(CONFIG["seed"])

    print("⏳ Memuat model CNN...")
    ocr_module.load_digit_model(CONFIG["model_path"], CONFIG["constants_path"])
    print("✅ Model berhasil dimuat.")

    test_dir = Path(CONFIG["test_dir"])
    gt_dir = Path(CONFIG["gt_dir"])

    images = sorted(
        p for p in test_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    results = []
    for img_path in images:
        gt_path = gt_dir / (img_path.stem + ".json")
        if not gt_path.exists():
            print(f"  (skip) {img_path.name}: Tidak ada file ground truth di {gt_path}")
            continue
        gt = json.loads(gt_path.read_text())
        try:
            out = run_pipeline(img_path, ocr_module)
            results.append((img_path, gt, out))
            print(f"  ✅ {img_path.name} — selesai ({out['elapsed']:.2f}s)")
        except Exception as e:
            print(f"  ❌ {img_path.name} — error: {e}")

    if not results:
        print(f"❌ Tidak ada data pengujian yang cocok di '{CONFIG['test_dir']}' dan '{CONFIG['gt_dir']}'.")
        return None

    # 1. Grid Detection
    grid_ok = 0
    for _, gt, out in results:
        if not out["used_perspective"] or out["corners"] is None:
            continue
        gt_quad = np.array(gt["corners"], dtype=np.float32)
        if polygon_iou(out["corners"], gt_quad, out["img_shape"]) >= CONFIG["iou_threshold"]:
            grid_ok += 1
    grid_detection_acc = 100.0 * grid_ok / len(results)

    # 2. Cell Segmentation IoU
    cell_ious = []
    for _, gt, out in results:
        if not out["used_perspective"] or out["corners"] is None:
            continue
        gt_cells = cell_polygons_from_quad(np.array(gt["corners"], dtype=np.float32))
        pred_cells = cell_polygons_from_quad(out["corners"])
        for key in gt_cells:
            cell_ious.append(polygon_iou(pred_cells[key], gt_cells[key], out["img_shape"]))
    cell_seg_iou = float(np.mean(cell_ious)) if cell_ious else 0.0

    # 3. Digit Recognition
    digit_accs = [clue_accuracy(out["board"], gt["board"]) for _, gt, out in results]
    digit_accs = [a for a in digit_accs if a is not None]
    digit_recognition_acc = float(np.mean(digit_accs)) if digit_accs else 0.0

    # 4. Sudoku Solving
    solved_correct = 0
    for _, gt, out in results:
        solved = solve_sudoku(out["board"])
        gt_solution = gt.get("solution") or solve_sudoku(gt["board"])
        if solved is not None and gt_solution is not None and solved == gt_solution:
            solved_correct += 1
    solving_acc = 100.0 * solved_correct / len(results)

    # 5. Processing Time
    times = [out["elapsed"] for _, _, out in results]
    avg_time = float(np.mean(times))
    median_time = float(np.median(times))

    # 6. Robustness
    drops = []
    for img_path, gt, out in results[: CONFIG["robustness_n"]]:
        base_acc = clue_accuracy(out["board"], gt["board"])
        if not base_acc:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        for degrade_fn in (low_light, extreme_angle):
            try:
                degraded_out = process_array(degrade_fn(img), ocr_module)
                deg_acc = clue_accuracy(degraded_out["board"], gt["board"]) or 0.0
                drops.append(max(0.0, base_acc - deg_acc))
            except Exception:
                drops.append(base_acc)
    robustness_drop = float(np.mean(drops)) if drops else None

    # Laporan 
    lines = [
        f"# Hasil Evaluasi Sistem OCR ({len(results)} gambar uji)\n",
        "| No | Metrik Evaluasi | Hasil Eksperimen | Target Metodologi | Status |",
        "|---|-----------------|------------------|-------------------|--------|",
        f"| 1 | Grid Detection Accuracy | {grid_detection_acc:.1f}% | >= 90% | {'Lolos' if grid_detection_acc >= 90 else 'Gagal'} |",
        f"| 2 | Cell Segmentation IoU | {cell_seg_iou:.3f} | >= 0.85 | {'Lolos' if cell_seg_iou >= 0.85 else 'Gagal'} |",
        f"| 3 | Digit Recognition Accuracy | {digit_recognition_acc:.1f}% | >= 90% | {'Lolos' if digit_recognition_acc >= 90 else 'Gagal'} |",
        f"| 4 | Sudoku Solving Accuracy | {solving_acc:.1f}% | >= 85% | {'Lolos' if solving_acc >= 85 else 'Gagal'} |",
        f"| 5 | Avg. Processing Time | {avg_time:.2f} detik (median {median_time:.2f}s) | <= 5 detik | {'Lolos' if avg_time <= 5 else 'Gagal'} |",
    ]
    if robustness_drop is not None:
        lines.append(
            f"| 6 | Robustness (Accuracy Drop) | {robustness_drop:.1f}% (Poin Absolut) | <= 15% | {'Lolos' if robustness_drop <= 15 else 'Gagal'} |"
        )
    else:
        lines.append("| 6 | Robustness (Accuracy Drop) | N/A | <= 15% | - |")

    report = "\n".join(lines)
    print("\n" + report + "\n")
    Path(CONFIG["output_file"]).write_text(report)
    print(f"Laporan disimpan ke: {CONFIG['output_file']}")

    return results


if __name__ == "__main__":
    results = main_eval()