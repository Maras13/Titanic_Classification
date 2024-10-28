
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier




def train_model(Xtrain_fe, y_train, Xtest_fe, y_test):

    model = [
    LogisticRegression(max_iter=400),
    DecisionTreeClassifier(),
    RandomForestClassifier(max_depth=6),
    xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    KNeighborsClassifier(3)
    
]
    
    results = {}
    
    for m in model:
        train_m = m.fit(Xtrain_fe, y_train)
        score = m.score(Xtrain_fe, y_train)
        
        results[type(m).__name__] = {"model": train_m,
                         "score": score
                        }
    
    return results, model



