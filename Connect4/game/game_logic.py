import game.constants as c
import numpy as np, math, logging, os
from game.board import Board
from ai_algorithms import alpha_beta as ap_dt, decisision_tree_connect4 as dt


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def human_move(bd: Board, board: np.ndarray, turn: int, game_mode: int, interface: any) -> bool:
	"""Set the column of human move"""
	print("\nEscolha uma coluna de 1 a 7: ", end='')
	col = int(input()) -1
	while not 0 <= col < 7:
		print("Escolha uma opção válida")
		col = int(input()) -1
	if not is_valid(board, col): return False 
	make_move(bd, board, turn, col, game_mode, interface)
	return True

def make_move(bd: Board, board: np.ndarray, turn: int, move: int, game_mode: int, interface: any):
	"""Make the move and see if the move is a winning one"""
	row = get_next_open_row(board, move)
	drop_piece(board, row, move, turn)	# adds the new piece to the data matrix
	os.system('clear')
	interface.print_game_modes(game_mode)
	bd.print_board()

	return winning_move(board, turn) or is_game_tied(board)


def ai_move(bd: Board, game_mode: int, board: np.ndarray, turn: int, interface: any) -> int:
	"""Set the column of the AI move"""
	ai_column = get_ai_column(board, game_mode)
	game_over = make_move(bd, board, turn, ai_column, game_mode, interface)
	return game_over
	

def get_ai_column(board: Board, game_mode: int) -> int:
	"""Select the chose ai algorithm to make a move"""
	#  1- Árvore de Decisão\n 2- AlphaBeta com Árvore de Decisão"
	chosen_column = 0
	if game_mode == 2:
		chosen_column = dt.decisiontree(board)
	elif game_mode == 3:
		chosen_column = ap_dt.alpha_beta(board)
	return chosen_column

def simulate_move(board: np.ndarray, piece: int, col: int) -> None | np.ndarray:
	"""Simulate a move in a copy of the board"""
	board_copy = board.copy()
	row = get_next_open_row(board_copy, col)
	if row == None: return None
	drop_piece(board_copy, row, col, piece)
	return board_copy

def get_next_open_row(board: np.ndarray, col: int) -> int:
	"""Given a column, return the first row avaiable to set a piece"""
	for row in range(c.ROWS):
		if board[row][col] == 0:
			return row
	return -1


def available_moves(board: np.ndarray) -> list | int:
	avaiable_moves = []
	for i in range(c.COLUMNS):
		if(board[5][i])==0:
			avaiable_moves.append(i)
	return avaiable_moves if len(avaiable_moves) > 0 else -1


def drop_piece(board: np.ndarray, row: int, col: int, piece: int) -> None:
	"""Insert a piece into board on correct location"""
	board[row][col] = piece


def is_game_tied(board: np.ndarray) -> bool:
	"""Assert if the game is tied"""
	for i in range(len(board)):
		for j in range(len(board[0])):
			if board[i][j]==0: return False
	return True


def is_valid(board: np.ndarray, col: int) -> bool:
	"""Analize if chosen column is valid"""
	if not 0 <= col < c.COLUMNS: return False
	row = get_next_open_row(board, col)
	return 0 <= row <= 5


def winning_move(board: np.ndarray, piece: int) -> bool:
	"""Return if the selected move will win the game"""
	def check_horizontal(board: np.ndarray, piece: int) -> bool:
		"""Check winning condition on horizontal lines"""
		for col in range(c.COLUMNS-3):
			for row in range(c.ROWS):
				if board[row][col] == piece and board[row][col+1] == piece and board[row][col+2] == piece and board[row][col+3] == piece:
					return True
			

	def check_vertical(board: np.ndarray, piece: int) -> bool:
		"""Check winning condition on vertical lines"""
		for col in range(c.COLUMNS):
			for row in range(c.ROWS-3):
				if board[row][col] == piece and board[row+1][col] == piece and board[row+2][col] == piece and board[row+3][col] == piece:
					return True
				

	def check_ascending_diagonal(board: np.ndarray, piece: int) -> bool:
		"""Check winning condition on ascending diagonal lines"""
		for col in range(c.COLUMNS-3):
			for row in range(c.ROWS-3):
				if board[row][col] == piece and board[row+1][col+1] == piece and board[row+2][col+2] == piece and board[row+3][col+3] == piece:
					return True
				

	def check_descending_diagonal(board: np.ndarray, piece: int) -> bool:
		"""Check winning condition on descending diagonal lines"""
		for col in range(c.COLUMNS-3):
			for row in range(3, c.ROWS):
				if board[row][col] == piece and board[row-1][col+1] == piece and board[row-2][col+2] == piece and board[row-3][col+3] == piece:
					return True
				
	return check_vertical(board, piece) or check_horizontal(board, piece) or check_ascending_diagonal(board, piece) or check_descending_diagonal(board, piece)
			
