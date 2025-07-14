import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, precision_recall_curve, auc


def run_cross_validation(X: pd.DataFrame, y: pd.Series, groups: pd.Series, model_class, model_params: dict, n_splits: int = 5):
    """
    Executa a validação cruzada para um dado modelo e retorna as métricas.

    Args:
        X (pd.DataFrame): DataFrame com as features.
        y (pd.Series): Series com a variável-alvo.
        groups (pd.Series): Series com os grupos para validação (ID_CLIENTE).
        model_class: A classe do modelo a ser treinado (ex: xgb.XGBClassifier).
        model_params (dict): Dicionário com os parâmetros do modelo.
        n_splits (int): Número de folds para a validação cruzada.

    Returns:
        dict: Um dicionário contendo as listas de scores para cada métrica.
    """
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=42)

    metrics = {'auc': [], 'recall': [], 'precision': [], 'f1': []}

    print(
        f"Iniciando validação cruzada para o modelo {model_class.__name__}...")
    for fold, (train_index, val_index) in enumerate(sgkf.split(X, y, groups=groups)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = model_class(**model_params)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred_class = model.predict(X_val)

        metrics['auc'].append(roc_auc_score(y_val, y_pred_proba))
        metrics['recall'].append(recall_score(y_val, y_pred_class))
        metrics['precision'].append(precision_score(
            y_val, y_pred_class, zero_division=0))
        metrics['f1'].append(f1_score(y_val, y_pred_class))

    print("Validação cruzada concluída.")
    return metrics


def find_optimal_threshold(model, X_val: pd.DataFrame, y_val: pd.Series):
    """
    Encontra o threshold de probabilidade que otimiza o F1-Score.

    Args:
        model: Um modelo já treinado.
        X_val (pd.DataFrame): Dados de validação para as features.
        y_val (pd.Series): Dados de validação para o alvo.

    Returns:
        float: O valor do threshold ótimo.
    """
    print("Encontrando o threshold ótimo...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)

    # Adicionamos um epsilon para evitar divisão por zero
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

    best_f1_idx = np.argmax(f1_scores[:-1])

    return thresholds[best_f1_idx]


def train_final_model(X: pd.DataFrame, y: pd.Series, model_class, model_params: dict):
    """
    Treina o modelo final com 100% dos dados de desenvolvimento.

    Args:
        X (pd.DataFrame): DataFrame completo de features.
        y (pd.Series): Series completa do alvo.
        model_class: A classe do modelo.
        model_params (dict): Os parâmetros do modelo.

    Returns:
        Um objeto de modelo treinado.
    """
    print("Treinando o modelo final com todos os dados...")
    final_model = model_class(**model_params)
    final_model.fit(X, y)
    print("Modelo final treinado com sucesso.")
    return final_model
