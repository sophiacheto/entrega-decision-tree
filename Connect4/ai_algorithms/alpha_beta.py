from game import constants as c, game_logic as game
from ai_algorithms import heuristic as h
from joblib import load, dump
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import random
from time import time


class DecisionTree:
    def __init__(self) -> None:
        self.dt = load('ai_algorithms/connect4_dt.joblib')
       

    def play(self, board):
        worst_moves = []
        available_moves = game.available_moves(board)
        
        boards = [game.simulate_move(board, c.AI_PIECE, col) for col in available_moves]
        rows = pd.concat([self.map_board_to_csv_row(b) for b in boards])
        
        predictions = self.dt.predict(rows)
        worst_moves = [col for col, pred in zip(available_moves, predictions) if pred == 'win']
        
        if worst_moves:
            return random.choice(worst_moves)
        

    def map_board_to_csv_row(self, board):
        flattened_board = np.array(board).flatten()
        converted_board = np.empty_like(flattened_board, dtype=str)

        # Use numpy masks to replace values
        converted_board[flattened_board == 0] = 'b'
        converted_board[flattened_board == 1] = 'x'
        converted_board[flattened_board == 2] = 'o'
        result = pd.DataFrame([converted_board])
        return result


def alpha_beta(board: np.ndarray):
    dt = DecisionTree()
    children = get_children(board, c.AI_PIECE)
    depth_limit = 6
    best_moves = []
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(calculate, child, 1, float('-inf'), float('+inf'), depth_limit, False, dt) for (child, col) in children]
        results = [(col, future.result()) for (child, col), future in zip(children, futures)]
    
    for col, score in results:
        if game.winning_move(board, c.AI_PIECE): return col
        if score == 1: best_moves.append(col)
    
    if best_moves:
        return random.choice(best_moves)
    else: return random.choice([col for (_, col) in children])
    
    

def calculate(board: np.ndarray, depth: int, alpha: int, beta: int, depth_limit: int, maximizing, dt):
    """Return the accumulated score for the current move"""

    if depth == depth_limit or game.winning_move(board, 1) or game.winning_move(board, 2) or game.is_game_tied(board):
        result = dt.play(board) 
        if result is None: return -1
        return 1
    
    if maximizing:
        maxEval = float('-inf')
        children = get_children(board, c.AI_PIECE)
        for (child, _) in children:
            eval = calculate(child, depth+1, alpha, beta, depth_limit, False, dt)
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval
    
    else:
        minEval = float('+inf')
        children = get_children(board, c.HUMAN_PIECE)
        for (child, _) in children:
            eval = calculate(child, depth+1, alpha, beta, depth_limit, True, dt)
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval


def get_children(board, piece) -> None:
    """Return children of the actual state board"""
    children = []
    if game.available_moves(board) == -1: return children
    for col in game.available_moves(board):  
        copy_board = game.simulate_move(board, piece, col)   
        children.append((copy_board, col)) 
    return children

