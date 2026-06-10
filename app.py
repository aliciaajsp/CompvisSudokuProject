import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

from ocr.ocr import load_digit_model, main as ocr_main
from solver.solver import SudokuSolver

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
TEST_DIR = PROJECT_ROOT / "test_images"

DEFAULT_MODEL = MODEL_DIR / "digit_cnn_tmnist.pt"
DEFAULT_CONSTANTS = MODEL_DIR / "norm_constants_tmnist.npy"

st.set_page_config(page_title="Sudoku CV", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🧩 Sudoku Computer Vision")

with st.sidebar:
    st.header("Input")
    use_sample = st.checkbox("Gunakan sample", value=True)
    sample_image = None
    uploaded_file = None
    camera_file = None
    if use_sample:
        sample_image = st.selectbox(
            "Pilih sample gambar",
            options=sorted([p.name for p in TEST_DIR.glob("*.png")]),
        )
    else:
        input_mode = st.radio("Sumber input", ["Upload file", "Kamera"], horizontal=True)
        if input_mode == "Upload file":
            uploaded_file = st.file_uploader("Upload gambar papan sudoku", type=["png", "jpg", "jpeg"])
        else:
            camera_file = st.camera_input("Ambil foto papan sudoku", label_visibility="collapsed")

image_path = None
if camera_file is not None:
    image_path = camera_file
elif uploaded_file is not None:
    image_path = uploaded_file
elif use_sample and sample_image is not None:
    image_path = str(TEST_DIR / sample_image)

model_path = str(DEFAULT_MODEL)
constants_path = str(DEFAULT_CONSTANTS)

@st.cache_resource
def load_model():
    return load_digit_model(model_path, constants_path)

if st.button("🔍 Deteksi & Selesaikan"):
    if image_path is None:
        st.warning("Pilih atau upload gambar terlebih dahulu.")
        st.stop()

    with st.spinner("Memproses..."):
        if hasattr(image_path, "read"):
            temp_path = str(PROJECT_ROOT / "temp_uploaded.png")
            with open(temp_path, "wb") as f:
                f.write(image_path.read())
            proc_path = temp_path
        else:
            proc_path = str(image_path)

        try:
            load_model()
            start = time.time()
            board = ocr_main(
                image_path=proc_path,
                model_path=model_path,
                constants_path=constants_path,
            )
            ocr_time = time.time() - start

            solver = SudokuSolver([row[:] for row in board])
            start = time.time()
            solved = solver.solve()
            solve_time = time.time() - start

            img = cv2.imread(proc_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Gambar Input")
                st.image(img, width=400)
            with c2:
                st.subheader("Status")
                st.session_state.history.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "image": sample_image or "Uploaded",
                    "original": [row[:] for row in board],
                    "solved": [row[:] for row in solver.board] if solved else None,
                    "ocr_time": ocr_time,
                    "solve_time": solve_time,
                    "status": "Terselesaikan" if solved else "Gagal diselesaikan"
                })
                st.session_state.history = st.session_state.history[-10:]
                if solved:
                    st.success(
                        f"Berhasil diselesaikan!\nOCR: {ocr_time:.2f}s | Solver: {solve_time:.4f}s"
                    )
                else:
                    st.error(
                        "Tidak ada solusi valid. Kemungkinan OCR error atau puzzle tidak valid."
                    )

            def draw_board(original, solved_board, title):
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.set_title(title)
                ax.axis("off")
                ax.set_xlim(0, 9)
                ax.set_ylim(0, 9)
                for r in range(9):
                    for c in range(9):
                        val = solved_board[r][c]
                        detected = original[r][c] != 0
                        color = "black" if detected else "red"
                        if val != 0:
                            ax.text(
                                c + 0.5,
                                8.5 - r,
                                str(val),
                                va="center",
                                ha="center",
                                fontsize=12,
                                color=color,
                                weight="bold",
                            )
                for i in range(10):
                    lw = 2 if i % 3 == 0 else 0.8
                    ax.plot([0, 9], [i, i], color="black", linewidth=lw)
                    ax.plot([i, i], [0, 9], color="black", linewidth=lw)
                return fig

            c3, c4 = st.columns(2)
            with c3:
                st.pyplot(draw_board(board, board, "Terdeteksi"))
            with c4:
                if solved:
                    st.pyplot(draw_board(board, solver.board, "Terselesaikan"))

        except Exception as e:
            st.exception(e)

with st.expander("History (max 10)"):
    if not st.session_state.history:
        st.write("Belum ada history")
    for h in reversed(st.session_state.history):
        st.markdown(f"**{h['timestamp']}** - {h['image']} | {h['status']} | OCR: {h['ocr_time']:.2f}s | Solver: {h['solve_time']:.4f}s")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Terdeteksi")
            st.code("\n".join([" ".join(map(str, row)) for row in h["original"]]), language=None)
        with c2:
            st.write("Terselesaikan")
            if h["solved"]:
                st.code("\n".join([" ".join(map(str, row)) for row in h["solved"]]), language=None)
            else:
                st.write("Tidak ada solusi valid. Kemungkinan OCR error atau puzzle tidak valid.")
        st.divider()