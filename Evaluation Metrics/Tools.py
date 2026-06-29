import cv2
import numpy as np
import json
import base64
import time
from pathlib import Path
import argparse
import ipywidgets as widgets
from google.colab import output
from IPython.display import HTML, display


def order_points(pts):
    """Mengurutkan 4 titik acak menjadi Top-Left, Top-Right, Bottom-Right, Bottom-Left"""
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def has_conflicts(board):
    """Validasi aturan Sudoku untuk mengecek salah ketik (typo)"""
    problems = []
    for r in range(9):
        seen = {}
        for c in range(9):
            v = board[r][c]
            if v == 0: continue
            if v in seen: problems.append(f"Baris {r}: angka {v} kembar di kolom {seen[v]} dan {c}")
            seen[v] = c
    for c in range(9):
        seen = {}
        for r in range(9):
            v = board[r][c]
            if v == 0: continue
            if v in seen: problems.append(f"Kolom {c}: angka {v} kembar di baris {seen[v]} dan {r}")
            seen[v] = r
    for br in range(3):
        for bc in range(3):
            seen = {}
            for r in range(br * 3, br * 3 + 3):
                for c in range(bc * 3, bc * 3 + 3):
                    v = board[r][c]
                    if v == 0: continue
                    if v in seen: problems.append(f"Kotak ({br},{bc}): angka {v} kembar")
                    seen[v] = (r, c)
    return problems

def cv2_to_base64(img):
    """Mengubah gambar OpenCV menjadi teks Base64 untuk dikonsumsi HTML5 Canvas"""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


def colab_pick_corners(img_base64):
    """Widget HTML5 Canvas untuk memilih 4 sudut di Google Colab"""
    html_code = f"""
    <div id="corner_container" style="font-family: Arial, sans-serif; background: #222; color: #fff; padding: 15px; border-radius: 8px; width: fit-content;">
        <h3>📍 Langkah 1: Klik 4 Sudut Terluar Sudoku</h3>
        <p style="font-size:12px; color:#aaa;">Klik secara berurutan pada 4 sudut boks Sudoku Anda. Jika salah klik, tekan tombol Reset.</p>
        <div style="position: relative; display: inline-block;">
            <canvas id="canvas" style="border: 2px solid #555; cursor: crosshair;"></canvas>
        </div>
        <br/><br/>
        <button id="btn_reset" style="background:#d9534f; color:white; border:none; padding:8px 15px; border-radius:4px; cursor:pointer;">Reset</button>
        <button id="btn_skip" style="background:#f0ad4e; color:white; border:none; padding:8px 15px; border-radius:4px; cursor:pointer; margin-left:10px;">Skip Gambar Ini</button>
        <button id="btn_submit" disabled style="background:#5cb85c; color:white; border:none; padding:8px 15px; border-radius:4px; cursor:not-allowed; margin-left:10px;">Konfirmasi Sudut</button>
    </div>

    <script>
    (function() {{
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();

        let points = [];
        let scale = 1.0;
        const maxDim = 700;

        img.onload = function() {{
            if (img.width > img.height) {{
                scale = maxDim / img.width;
            }} else {{
                scale = maxDim / img.height;
            }}
            if (scale > 1.0) scale = 1.0;

            canvas.width = img.width * scale;
            canvas.height = img.height * scale;
            draw();
        }};
        img.src = "data:image/png;base64,{img_base64}";

        function draw() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            // Gambar Garis Penghubung
            if (points.length > 0) {{
                ctx.strokeStyle = '#00ff00';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(points[0].x, points[0].y);
                for (let i = 1; i < points.length; i++) {{
                    ctx.lineTo(points[i].x, points[i].y);
                }}
                if (points.length === 4) ctx.closePath();
                ctx.stroke();
            }}

            // Gambar Titik Merah
            points.forEach((p, idx) => {{
                ctx.fillStyle = '#ff0000';
                ctx.beginPath();
                ctx.arc(p.x, p.y, 6, 0, 2 * Math.PI);
                ctx.fill();

                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 14px Arial';
                ctx.fillText(idx + 1, p.x + 8, p.y - 8);
            }});
        }}

        canvas.addEventListener('click', function(e) {{
            if (points.length < 4) {{
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                points.push({{x: x, y: y}});
                draw();

                if (points.length === 4) {{
                    document.getElementById('btn_submit').disabled = false;
                    document.getElementById('btn_submit').style.cursor = 'pointer';
                }}
            }}
        }});

        document.getElementById('btn_reset').addEventListener('click', function() {{
            points = [];
            document.getElementById('btn_submit').disabled = true;
            document.getElementById('btn_submit').style.cursor = 'not-allowed';
            draw();
        }});

        document.getElementById('btn_skip').addEventListener('click', function() {{
            google.colab.kernel.invokeFunction('notebook.callback_corners', ['skip', []], {{}});
        }});

        document.getElementById('btn_submit').addEventListener('click', function() {{
            const origPoints = points.map(p => [p.x / scale, p.y / scale]);
            google.colab.kernel.invokeFunction('notebook.callback_corners', ['confirm', origPoints], {{}});
        }});
    }})();
    </script>
    """
    display(HTML(html_code))

def colab_pick_board(warped_base64):
    """Widget Grid 9x9 interaktif untuk mengisi angka Sudoku menggunakan keyboard di Colab"""
    html_code = f"""
    <div id="board_container" style="font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; padding: 20px; border-radius: 8px; width: fit-content; margin-top:20px;">
        <h3>🔢 Langkah 2: Pengisian Matriks Angka Papan Sudoku</h3>
        <p style="font-size:12px; color:#aaa;">Klik kotak biru, lalu ketik angka [1-9]. Tekan [0] atau [Space] untuk mengosongkan. Gunakan tombol panah keyboard untuk berpindah.</p>

        <div style="display: flex; gap: 20px; align-items: flex-start;">
            <div style="position: relative;">
                <canvas id="board_canvas" width="540" height="540" style="border: 2px solid #fff;"></canvas>
            </div>
            <div style="background:#2a2a2a; padding:15px; border-radius:6px; width:220px;">
                <h4 style="margin-top:0;">Kontrol Navigasi:</h4>
                <ul style="font-size:12px; padding-left:20px; color:#ccc; line-height:1.6;">
                    <li><b>Klik Mouse</b>: Pilih Sel</li>
                    <li><b>Angka 1-9</b>: Isi & Otomatis Maju</li>
                    <li><b>Angka 0 / Space</b>: Kosongkan Sel</li>
                    <li><b>Backspace</b>: Hapus angka</li>
                    <li><b>Tombol Panah</b>: Geser Sel</li>
                </ul>
                <hr style="border-color:#444;"/>
                <button id="btn_save_board" style="background:#5cb85c; color:white; border:none; padding:10px 15px; border-radius:4px; cursor:pointer; width:100%; font-weight:bold;">💾 SIMPAN DATA (JSON)</button>
                <button id="btn_skip_board" style="background:#f0ad4e; color:white; border:none; padding:8px 15px; border-radius:4px; cursor:pointer; width:100%; margin-top:10px;">Skip Gambar</button>
            </div>
        </div>
    </div>

    <script>
    (function() {{
        const canvas = document.getElementById('board_canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();

        let board = Array(9).fill(0).map(() => Array(9).fill(0));
        let curRow = 0;
        let curCol = 0;
        const cellSize = 60; // 540 / 9

        img.onload = function() {{
            draw();
        }};
        img.src = "data:image/png;base64,{warped_base64}";

        function draw() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            // Gambar Garis Grid Sudoku
            for (let i = 0; i <= 9; i++) {{
                ctx.strokeStyle = (i % 3 === 0) ? '#00ff00' : '#00aa00';
                ctx.lineWidth = (i % 3 === 0) ? 3 : 1;

                // Horizontal
                ctx.beginPath(); ctx.moveTo(0, i * cellSize); ctx.lineTo(canvas.width, i * cellSize); ctx.stroke();
                // Vertical
                ctx.beginPath(); ctx.moveTo(i * cellSize, 0); ctx.lineTo(i * cellSize, canvas.height); ctx.stroke();
            }}

            // Gambar Angka yang diinput
            for (let r = 0; r < 9; r++) {{
                for (let c = 0; c < 9; c++) {{
                    if (board[r][c] !== 0) {{
                        ctx.fillStyle = '#ff3333';
                        ctx.font = 'bold 26px Arial';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillText(board[r][c], c * cellSize + cellSize/2, r * cellSize + cellSize/2);
                    }}
                }}
            }}

            // Gambar Kotak Kursor Aktif
            ctx.strokeStyle = '#00c3ff';
            ctx.lineWidth = 4;
            ctx.strokeRect(curCol * cellSize, curRow * cellSize, cellSize, cellSize);
        }}

        canvas.addEventListener('click', function(e) {{
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            curRow = Math.min(8, Math.max(0, Math.floor(y / cellSize)));
            curCol = Math.min(8, Math.max(0, Math.floor(x / cellSize)));
            draw();
        }});

        function advance() {{
            curCol++;
            if (curCol > 8) {{
                curCol = 0;
                curRow = (curRow + 1) % 9;
            }}
        }}

        window.addEventListener('keydown', function(e) {{
            if(["Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.code)) {{
                e.preventDefault();
            }}

            if (e.key >= '1' && e.key <= '9') {{
                board[curRow][curCol] = parseInt(e.key);
                advance();
                draw();
            }} else if (e.key === '0' || e.key === ' ') {{
                board[curRow][curCol] = 0;
                advance();
                draw();
            }} else if (e.key === 'Backspace') {{
                board[curRow][curCol] = 0;
                draw();
            }} else if (e.key === 'ArrowLeft') {{
                curCol = Math.max(0, curCol - 1); draw();
            }} else if (e.key === 'ArrowRight') {{
                curCol = Math.min(8, curCol + 1); draw();
            }} else if (e.key === 'ArrowUp') {{
                curRow = Math.max(0, curRow - 1); draw();
            }} else if (e.key === 'ArrowDown') {{
                curRow = Math.min(8, curRow + 1); draw();
            }}
        }});

        document.getElementById('btn_save_board').addEventListener('click', function() {{
            google.colab.kernel.invokeFunction('notebook.callback_board', ['confirm', board], {{}});
        }});

        document.getElementById('btn_skip_board').addEventListener('click', function() {{
            google.colab.kernel.invokeFunction('notebook.callback_board', ['skip', []], {{}});
        }});
    }})();
    </script>
    """
    display(HTML(html_code))


corner_result = {"action": None, "data": None}
board_result = {"action": None, "data": None}

def main_pipeline(test_dir="test_images", gt_dir="ground_truth", warp_size=900):
    test_path = Path(test_dir)
    gt_path = Path(gt_dir)
    gt_path.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in test_path.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    todo = [p for p in images if not (gt_path / (p.stem + ".json")).exists()]

    print(f"📋 Ditemukan {len(images)} gambar, {len(todo)} gambar belum diberi label.\n")
    if not todo:
        print("✅ Semua gambar sudah selesai dilabeli!")
        return

    out_widget = widgets.Output()
    display(out_widget)

    state = {"current_idx": 0}

    def handle_corners(action, pts):
        corner_result["action"] = action
        corner_result["data"] = pts

        with out_widget:
            if action == "skip":
                print(f"⚠️ Gambar {todo[state['current_idx']].name} dilewati.")
                next_image()
            elif action == "confirm":
                output.clear(output_tags='corner_container')
                process_board(pts)

    def handle_board(action, matrix):
        board_result["action"] = action
        board_result["data"] = matrix

        with out_widget:
            output.clear(output_tags='board_container')
            if action == "confirm":
                img_file = todo[state['current_idx']]
                ordered_corners = order_points(corner_result["data"])


                out_data = {
                    "corners": ordered_corners.tolist(),
                    "board": matrix
                }
                json_file_path = gt_path / (img_file.stem + ".json")
                json_file_path.write_text(json.dumps(out_data, indent=2))
                print(f"💾 BERHASIL DISIMPAN: {json_file_path.name}")

                problems = has_conflicts(matrix)
                if problems:
                    print(f"🚨 PERINGATAN: Ada {len(problems)} konflik angka.")

            next_image()


    output.register_callback('notebook.callback_corners', handle_corners)
    output.register_callback('notebook.callback_board', handle_board)

    def process_board(pts_clicked):
        img_file = todo[state['current_idx']]
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        ordered_corners = order_points(pts_clicked)

        dst = np.array([[0, 0], [warp_size, 0], [warp_size, warp_size], [0, warp_size]], dtype="float32")
        M = cv2.getPerspectiveTransform(ordered_corners, dst)
        warped = cv2.warpPerspective(img, M, (warp_size, warp_size))

        colab_pick_board(cv2_to_base64(warped))

    def start_current_image():
        if state["current_idx"] >= len(todo):
            with out_widget:
                print("\n🏁 Selesai! Semua antrean gambar berhasil diproses.")
            return

        img_file = todo[state["current_idx"]]
        with out_widget:
            print(f"\n========================================\n▶️ Memproses Gambar ({state['current_idx']+1}/{len(todo)}): {img_file.name}")

        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            with out_widget:
                print(f"❌ Gagal membaca gambar {img_file.name}. Lanjut ke berikutnya.")
            next_image()
            return


        colab_pick_corners(cv2_to_base64(img))

    def next_image():
        state["current_idx"] += 1
        start_current_image()


    start_current_image()


main_pipeline(test_dir="test_images", gt_dir="ground_truth")