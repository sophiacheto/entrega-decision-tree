from classifier.node import Node
from pandas import DataFrame, Series
import numpy as np
import pandas as pd

class DecisionTree:
    def __init__(self, dataset: DataFrame, min_samples: int = 2) -> None:
        self.root: Node = None
        self.dataset: DataFrame = dataset
        self.min_samples: int = min_samples       
        self.feature_types = dict(map(self.attr_mapping, zip(dataset.columns, dataset.dtypes)))


    def attr_mapping(self,tuplo):
        if tuplo[1] in [np.int64, np.float64]: return (tuplo[0], 'continuous')
        return (tuplo[0],'discrete')
    

    def __str__(self) -> str:
        return self.toString(self.root, "")


    def fit(self, dataset: DataFrame) -> None:
        self.root = self.build_tree(dataset)


    def predict(self, X_test: DataFrame) -> list:
        return [self.make_prediction(row, self.root) for _, row in X_test.iterrows()]
    

    def build_tree(self, dataset: DataFrame) -> Node:
        features = dataset.iloc[:,:-1]
        targets = dataset.iloc[:,-1]
        num_samples = len(dataset)
        ispure = (len(set(targets)) == 1)
        children = []

        if num_samples < self.min_samples or ispure or len(features.columns)==0 or num_samples==0:  
            return Node(value=self.calculate_leaf_value(targets), is_leaf=True, dataset=dataset)
        
        best_split = self.get_best_split(dataset)
        if best_split == {}: 
            return Node(value=self.calculate_leaf_value(targets), is_leaf=True, dataset=dataset)

        for child in best_split["children"]:
            if len(child) == 0: children.append(Node(value=self.calculate_leaf_value(targets), 
                                                     is_leaf=True, dataset=child))
            else: children.append(self.build_tree(child))

        return Node(dataset, children, best_split["value"], best_split["info_gain"], 
                    best_split["feature_name"], best_split["split_type"])
    

    def get_best_split(self, dataset: DataFrame) -> dict:
        best_split = {}
        max_infogain = 0
        features = dataset.iloc[:,:-1]
        targets = dataset.iloc[:, -1]
        parent_entropy = self.entropy(targets) 

        for feature_name in features.columns:
            values = self.dataset[feature_name]
            feature_type = self.feature_types[feature_name]
            
            if feature_type == 'discrete':
                children, info_gain = self.discrete_split(dataset, feature_name, pd.unique(values), targets, parent_entropy)
                max_infogain = self.update_best_split(best_split, info_gain, max_infogain, feature_type, children, pd.unique(values.map(lambda x: str(x))), feature_name)
                continue

            for value in pd.unique(values):
                children, info_gain = self.continuous_split(dataset, feature_name, value, targets, parent_entropy)
                max_infogain = self.update_best_split(best_split, info_gain, max_infogain, feature_type, children, value, feature_name)
        
        return best_split
    

    def update_best_split(self, best_split: dict, info_gain: float, max_infogain: float, feature_type: str, children: list, value: any, feature_name: str) -> float:
        if info_gain > max_infogain:
            best_split["feature_name"] = feature_name
            best_split["value"] = value
            best_split["split_type"] = feature_type
            best_split["children"] = children
            best_split["info_gain"] = info_gain
            return info_gain
        return max_infogain
    

    def continuous_split(self, dataset: DataFrame, feature_name, threshold, targets, parent_entropy) -> tuple[Node, float]:
        left = dataset[dataset[feature_name] <= threshold].copy().drop([feature_name], axis=1)
        right = dataset[dataset[feature_name] > threshold].copy().drop([feature_name], axis=1)
        children = [left, right]
        info_gain = self.info_gain(children, targets.shape[0], parent_entropy)
        return children, info_gain
    
    
    def discrete_split(self, dataset: DataFrame, feature_name, values, targets, parent_entropy) -> tuple[Node, float]:
        children = [(dataset[dataset[feature_name] == label].copy()).drop([feature_name], axis=1) for label in values]
        info_gain = self.info_gain(children, targets.shape[0], parent_entropy)
        return children, info_gain
    
    
    def info_gain(self, children: list[DataFrame], parent_length: int, parent_entropy) -> float:
        children_entropy = np.sum([(len(child_dataset) / parent_length) * self.entropy(child_dataset.iloc[:, -1]) for child_dataset in children])
        return parent_entropy - children_entropy


    def entropy(self, targets: Series) -> float:
        counts = targets.value_counts()
        probs = counts/len(targets)
        return np.sum(-probs * np.log2(probs))
    

    def calculate_leaf_value(self, targets: DataFrame) -> any:
        targets_list = list(targets)
        counting = [(targets_list.count(item), item) for item in set(targets_list)]
        return max(counting)[1]
    

    def make_prediction(self, row: tuple, node: Node) -> any:
        if node.is_leaf: 
            return node.value
        
        value = row[node.feature_name]

        if node.split_type == 'discrete': 
            for i, node_value in enumerate(node.value):
                if str(value) == node_value:
                    return self.make_prediction(row, node.children[i])  
        
        if value <= node.value:
            return self.make_prediction(row, node.children[0])  
        return self.make_prediction(row, node.children[1])
    


    def toString(self, node: Node, indent: str) -> str:
        string = ""
        if not node.children:
            return string
        
        
        add = " " * 5
        indent += add

        for i in range(len(node.children)):
            child = node.children[i]

            if type(node.value) in [np.int64, np.float64]:
                if i==0: simbolo="<="
                else: simbolo=">"
                if child.is_leaf:
                    string += indent + f"\"{node.feature_name}\" {simbolo} {node.value}: {child.value} ({child.size})" + "\n"
                else:
                    string += indent + f"\"{node.feature_name}\" {simbolo} {node.value}:" + "\n"
                    string += self.toString(child, indent+add)
            else: 
                if child.is_leaf: 
                    string += indent + f"\"{node.feature_name}\" {node.value[i]}: {child.value} ({child.size})" + "\n"  
                else:
                    string += indent + f"\"{node.feature_name}\" {node.value[i]}:" + "\n"
                    string += self.toString(child, indent+add)
        
        return string


       
   

       