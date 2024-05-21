from typing import Union
import numpy as np
import pandas as pd
import random
from pandas import DataFrame, Series, read_csv
from game import game_logic as game
from game import constants as c
from joblib import load, dump

been_called = False

class DTNode:
    def __init__(self, feature_index=None, feature_name=None, children=None, info_gain=None, split_values = None, leaf_value=None) -> None:
        self.feature_index = feature_index   
        self.feature_name = feature_name      
        self.children = children
        self.info_gain = info_gain 
        self.split_values = split_values 
        self.leaf_value = leaf_value 


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = None, min_samples_split: int = None, criterium: str = 'entropy') -> None:
        self.root: DTNode = None
        self.max_depth: int = max_depth 
        self.min_samples_split: int = min_samples_split 
        self.criterium: str = criterium


    def predict(self, X_test: DataFrame) -> list:
        '''Predict target column for a dataframe'''
        return [self.make_prediction(row, self.root) for _, row in X_test.iterrows()]



    def make_prediction(self, row: tuple, node: DTNode) -> Union[any, None]:
        '''Predict target for each row in dataframe'''
        if node.leaf_value is not None: 
            return node.leaf_value
        
        index = node.feature_index
        attribute = node.feature_name
        value = row[index]

        for i, node_value in enumerate(node.split_values):
            if value == node_value:
                return self.make_prediction(row, node.children[i])  
        return None
    


class DecisionTree:

    def __init__(self) -> None:
        self.dt = load('ai_algorithms/connect4_dt.joblib')
       

    def play(self, board):
        best_moves = []
        average_moves = []
        worst_moves = []
        available_moves = game.available_moves(board)  # Assuming game is an instance of some class

        for col in available_moves:
            temp_board = game.simulate_move(board, c.AI_PIECE, col)  # Assuming c.AI_PIECE is defined
            row = self.map_board_to_csv_row(temp_board)  # Assuming this method is defined
            predict = self.dt.predict(row)
            if predict[0] == 'win': worst_moves.append(col)
            elif predict[0] == 'draw': average_moves.append(col)
            elif predict[0] == 'loss': best_moves.append(col)

        if best_moves:
            return random.choice(best_moves)
        elif average_moves:
            return random.choice(average_moves) 
        elif worst_moves:
            return random.choice(worst_moves)



    def map_board_to_csv_row(self, board):
        flattened_list = [item for sublist in board for item in sublist]
        result = pd.DataFrame([flattened_list])
        result.replace({0: 'b', 1: 'x', 2: 'o'}, inplace = True)
        return result


def decisiontree(board):
    dt = DecisionTree()
    return dt.play(board) 