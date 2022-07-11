import pandas as pd 
import fire

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score

from utils.preprocessing import clean_data
from utils.preprocessing import add_features
from utils.preprocessing import split_data
from utils.preprocessing import feature_importances_analysis
from utils.preprocessing import importance_feature
from utils.preprocessing import processed_df
from utils.preprocessing import actual_cat_features

from train.train import grid_searching
from train.train import train_model

from train.evaulating import prediction_model

def main(
    data_path: str='data/monitorings.csv',
    depth_range: list=[5,7],
    learning_rate_range: list=[0.1, 0.01],
    iterations_range: list=[400, 900],
    metrics_path: str='train/metrics.csv'
    ):

    monitorings = pd.read_csv(data_path)

    # Очистим данные от пропусков, дубликатов и лишних столбцов
    monitorings = clean_data(monitorings)

    # Добавим новые столбцы
    monitorings = add_features(monitorings)
        
    # Проанализируем важность каждого признака
    model_for_feature, feature_names = feature_importances_analysis(monitorings)

    # Получим значения влияния каждого признака
    score_data = importance_feature(model_for_feature, feature_names)

    # Удалим невлияющие признаки
    monitorings = processed_df(monitorings, score_data)

    # Получим актуальный список категориальных признаков
    category_features = actual_cat_features(monitorings)
        
    # Подберём лучшие параметры для модели
    grid = {
        'depth': depth_range,
        'learning_rate' : learning_rate_range,
        'iterations' : iterations_range
        }
        
    # Получаем лучшие параметры для обучения модели
    params = grid_searching(monitorings, grid, category_features)
        
    # Обучим модель
    model = train_model(monitorings, params, category_features)
        
    # Посчитаем метрики
    metrics = prediction_model(model, monitorings)
    pd.DataFrame([metrics]).to_csv(metrics_path)

if __name__ == '__main__':
    fire.Fire(main)