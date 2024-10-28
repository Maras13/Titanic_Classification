import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# data pre-processing stack
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer






class readPrepare:
    def __init__(self, file_path="./data/train.csv"):
        self.file_path = file_path
        self.feature_transform = None
        
        
    def read_split(self):
        df = pd.read_csv(self.file_path)

        df.drop(columns=['Cabin','Ticket','PassengerId'], inplace=True)

        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

        X = df_train.drop('Survived', axis=1)
        y = df_train['Survived']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
        
        return X_train, X_test, y_train, y_test





    def prepare_data(self, X_train, X_test):
        
        def name_length(df):
            length = df[df.columns[0]].str.len()
            return length.values.reshape(-1, 1)

        
        cat_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"),
                            OneHotEncoder(handle_unknown='ignore', sparse=False))
        num_pipe = make_pipeline(SimpleImputer(strategy='median'),
                            MinMaxScaler())
        # Define ColumnTransformer
        self.feature_transform  = ColumnTransformer(
            transformers=[
                ("num", num_pipe, ['Age', 'Fare']),
                ("cat", cat_pipe, ['Pclass', 'Embarked', 'Parch', 'Sex']),
                ("name", FunctionTransformer(name_length), ['Name']),
                ("do_nothing", 'passthrough', ['SibSp'])
            ]
        )

        self.feature_transform.fit(X_train)
        # Extract feature names
        
        feature_names = []

        for name, trans, columns in self.feature_transform.transformers_:
            if name == 'cat':
                feature_names.extend(trans.get_feature_names_out(input_features=columns))
        else:
            feature_names.extend(columns)
            
        Xtrain_fe = self.feature_transform.transform(X_train) 
        Xtest_fe = self.feature_transform.transform(X_test)
        
        return Xtrain_fe, Xtest_fe

















