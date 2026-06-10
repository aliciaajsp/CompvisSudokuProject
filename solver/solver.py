from ocr.ocr import main
import time


class SudokuSolver:
    def __init__(self, board):
        self.board = board

    def find_empty(self):
        for row in range(9):
            for col in range(9):
                if self.board[row][col] == 0:
                    return row, col
        return None

    def is_valid(self, num, row, col):

        # Check row
        for c in range(9):
            if self.board[row][c] == num:
                return False

        # Check column
        for r in range(9):
            if self.board[r][col] == num:
                return False

        # Check 3x3 box
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3

        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if self.board[r][c] == num:
                    return False

        return True

    def solve(self):
        empty = self.find_empty()

        if empty is None:
            return True

        row, col = empty

        for num in range(1, 10):

            if self.is_valid(num, row, col):

                self.board[row][col] = num

                if self.solve():
                    return True

                self.board[row][col] = 0

        return False

    def print_board(self):

        for i in range(9):

            if i % 3 == 0:
                print("+-------+-------+-------+")

            for j in range(9):

                if j % 3 == 0:
                    print("| ", end="")

                value = "." if self.board[i][j] == 0 else str(self.board[i][j])

                print(value, end=" ")

            print("|")

        print("+-------+-------+-------+")


if __name__ == "__main__":

    print("Reading Sudoku image...")

    board = main(
        image_path="test_images/sample2.png",
        model_path="model/digit_cnn_tmnist.pt",
        constants_path="model/norm_constants_tmnist.npy"
    )

    solver = SudokuSolver(board)

    print("\nDetected Sudoku:")
    solver.print_board()

    start = time.time()

    if solver.solve():

        elapsed = time.time() - start

        print("\nSolved Sudoku:")
        solver.print_board()

        print(f"\nSolved in {elapsed:.6f} seconds")

    else:

        print("\nNo valid solution exists.")
        print("Possible OCR error or invalid Sudoku puzzle.")