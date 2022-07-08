import pandas as pd

from utils.preprocessing import split_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score

def prediction_model(model, df):
    
    x_train, x_test, y_train, y_test = split_data(df)
    
    y_pred_prob = model.predict_proba(x_test)
    y_pred = y_pred_prob[:, 1] > 0.73
    
    f1 = f1_score(y_pred, y_test)
    presicion = precision_score(y_pred, y_test)
    recall = recall_score(y_pred, y_test)
    
    metrics = {
        'f1': f1,
        'precision': presicion,
        'recall': recall,
    }
    
    return metrics