import itertools, logging, os
from game import constants as c
from game import game_logic as game
from game.board import Board
from dataclasses import dataclass

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class Interface:
    rows: int = c.ROWS
    columns: int = c.COLUMNS

    def print_game_modes(self, value: int) -> None:
        game_modes = {1: 'Human x Human',
                      2: 'Árvore de Decisão',
                      3: 'AlphaBeta com Árvore de Decisão',}
        print(f"Modo de jogo escolhido: {game_modes[value]}\n")

    def start_game(self, bd: Board) -> None:
        """Set up the conditions to the game, as choose game_mode and draw the pygame display"""
        
        game_mode = int(input("Selecione um modo de jogo:\n 1- Árvore de Decisão\n 2- AlphaBeta com Árvore de Decisão\n")) +1
        os.system('clear')
        self.print_game_modes(game_mode)
        bd.print_board()
        self.play_game(bd, game_mode)


    def play_game(self, bd: Board, game_mode: int) -> None:
        """Run the game"""
        board = bd.get_board()	
        game_over = False
        turns = itertools.cycle([1, 2])  
        turn = next(turns)

        while not game_over:
            if turn == 1 or (turn == 2 and game_mode == 1):  # get human move
                if not game.human_move(bd, board, turn, game_mode, self): continue  # make a move
                if game.winning_move(board, turn): 
                    game_over = True
                    break
                turn = next(turns)

            if turn != 1 and game_mode != 1: 
                game_over = game.ai_move(bd, game_mode, board, turn, self)
                if game_over: break     
                turn = next(turns)

            # Evita que a ultima jogada no ultimo ponto possível retorne empate ao invès de vitória
            if game.is_game_tied(board) and game_over == False:
                print(f"Empate!")
                break   

        if not game.is_game_tied(board):
            print(f"Player {turn} venceu o jogo!")

  