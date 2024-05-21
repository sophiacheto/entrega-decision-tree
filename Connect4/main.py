from game.board import Board
from interface.interface import Interface

def main() -> None:
    board = Board()
    interface = Interface()
    interface.start_game(board)
    
if __name__ == "__main__":
    main()