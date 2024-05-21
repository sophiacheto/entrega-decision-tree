from sklearn.model_selection import LeaveOneOut, KFold
from classifier.decision_tree import DecisionTree
from sklearn.metrics import accuracy_score
from pandas import DataFrame
import pandas as pd
import numpy as np


class StatisticalAnalysis:
    def __init__(self, dataframe: DataFrame) -> None:
        self.df = dataframe
        self.target_data = self.df.iloc[:,-1]
        self.features_data = self.df.iloc[:,:-1]
        self.analysis()


    def analysis(self):
        if len(self.df)<50:
            accuracies, test_size = self._leave_one_out_cross_validation()
            print(f"\nCross validation type: Leave One Out")
        else:
            accuracies, test_size = self._k_fold_cross_validation()
            print(f"\nCross validation type: K-Fold")
        mean_accuracy = np.mean(accuracies)
        self._print_statistics(mean_accuracy, test_size)

        # accuracies, test_size = self.general_analysis()
        # mean_accuracy = np.mean(accuracies)
        # self.print_analysis(mean_accuracy, test_size)

        
    def _leave_one_out_cross_validation(self) -> tuple[list, int]:
        loo = LeaveOneOut()
        accuracies = []
        for train_index, test_index in loo.split(self.features_data):
            X_train, X_test = self.features_data.iloc[train_index], self.features_data.iloc[test_index]
            y_train, y_test = self.target_data.iloc[train_index], self.target_data.iloc[test_index]
            
            dt = DecisionTree(self.df)
            df = pd.concat([X_train, y_train], axis=1)
            dt.fit(df)
            y_pred = dt.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred)) 

        return accuracies, y_test.shape[0]
    
    
    def _k_fold_cross_validation(self, n_test: int = 10) -> tuple[list, int]:
        kf = KFold(n_splits=n_test)
        accuracies = []

        for train_index, test_index in kf.split(self.features_data):
            X_train, X_test = self.features_data.iloc[train_index], self.features_data.iloc[test_index]
            y_train, y_test = self.target_data.iloc[train_index], self.target_data.iloc[test_index]

            dt = DecisionTree(self.df)
            df = pd.concat([X_train, y_train], axis=1)
            dt.fit(df)
            y_pred = dt.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))

        return accuracies, y_test.shape[0]
    

    def _print_statistics(self, mean_accuracy: float, test_size: int) -> None:
        print(f"Model test size: {test_size} rows")
        print(f"Model train size: {len(self.df) - test_size} rows")
        print(f"Model accuracy: {(mean_accuracy * 100):.2f}%\n\n" )


    # Printar apenas cross validation
    # def print_analysis(self, mean_accuracy, test_size):
    #     print("GENERAL:")
    #     print(f"Model test size: {test_size} rows")
    #     print(f"Model train size: {len(self.df) - test_size} rows")
    #     print(f"Model accuracy: {(mean_accuracy * 100):.2f}%\n\n" )


    # def general_analysis(self):
    #     accuracies = []
    #     dt = DecisionTree(self.df)
    #     for i in range (35,45):
    #         train, test = train_test_split(self.df, test_size=0.3, random_state=i)
    #         dt = DecisionTree(self.df)
    #         dt.fit(train)

    #         target_test = test.iloc[:,-1]
    #         predictions = dt.predict(test)
    #         accuracies.append(accuracy_score(target_test, predictions))
    #     return accuracies, target_test.shape[0]