import numpy as np
from game import constants as c
from dataclasses import dataclass

@dataclass
class Board:
    rows: int = c.ROWS
    columns: int = c.COLUMNS
    board: np.ndarray = np.zeros((rows, columns))
            
    def get_board(self) -> np.ndarray:
        return self.board

    def print_board(self) -> None:
        print(np.flip(self.board, 0), "\n")
        
 