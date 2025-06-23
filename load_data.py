# load_data.py
import pandas as pd

def load_titanic_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df = df[features + ['Survived']].dropna()
    X = df[features]
    y = df['Survived']
    return X, y