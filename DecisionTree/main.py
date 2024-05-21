from classifier.statistics import StatisticalAnalysis as stats
from classifier.decision_tree import DecisionTree
from classifier import utils
from joblib import load
import pandas as pd
import time

IRIS_CSV = 'csv_files/iris.csv'
RESTAURANT_CSV = 'csv_files/restaurant.csv'
WEATHER_CSV = 'csv_files/weather.csv'
CONNECT4_CSV = 'csv_files/connect4.csv'


def main():
    chose_csv = _print_options()
    df = pd.read_csv(chose_csv)

    
    if chose_csv == CONNECT4_CSV:
        dt = load('classifier/dt_connect4.joblib')
    else: 
        df.drop(['ID'], axis=1, inplace=True)
        start = time.time()
        dt = DecisionTree(dataset=df)
        dt.fit(df)
        end = time.time()
        print(f'\nFit time: {end-start:.2f} seconds',)
        stats(df)
        
    target = df.iloc[:,-1]
    colors = {key:value for (value, key) in zip(["#bad9d3", "#d4b4dd", "#fdd9d9"], pd.unique(target))}
    utils.make_dot_representation(dt, colors)

    print("========== TREE ==========")
    print(dt)
    utils.predict(dt, df)


def _print_options() -> None:
    csvs = {1: 'csv_files/iris.csv',
            2: 'csv_files/restaurant.csv',
            3: 'csv_files/weather.csv',
            4: 'csv_files/connect4.csv'}
    
    print("Choose the dataset to train the Decision Tree:"
            "\n1 - Iris.csv\n"
            "2 - Restaurant.csv\n"
            "3 - Weather.csv\n"
            "4 - Connect4.csv\n")
    chose_csv = int(input("Dataset escolhido: "))
    return csvs[chose_csv]


if __name__ == '__main__':
    main()


