# preprocessor.py
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def get_preprocessor():
    num_features = ['Age', 'SibSp', 'Parch', 'Fare']
    cat_features = ['Pclass', 'Sex', 'Embarked']
    
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])
    return preprocessor
