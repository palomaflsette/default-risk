"""
Versão simples de um Pipeline de Treinamento completo para o case
"""

import pandas as pd
import numpy as np
import warnings
import sys
import os
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

current_dir = os.getcwd()
if 'src' not in sys.path:
    sys.path.append(current_dir)

try:
    from src.modeling import train_final_model
    from src.feature_engineering import create_advanced_features
    from src.data_processing import load_and_clean_data, load_data_and_setup_env
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print("Certifique-se de que está executando a partir da raiz do projeto")
    sys.exit(1)

warnings.filterwarnings('ignore')


def main():
    """Pipeline principal de treinamento e predição"""

    print("~*~ INICIANDO PIPELINE DE RISCO DE CRÉDITO ~*~")
    print("=" * 50)

    PATH_RAW = os.path.join("data", "raw")
    _, _, _, path_processed, _, _, _, _ = load_data_and_setup_env(
        current_dir)

    # Hiperparâmetros otimizados
    best_params = {
        'subsample': 0.6,
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.01,
        'gamma': 0.1,
        'colsample_bytree': 0.8
    }
    OPTIMAL_THRESHOLD = 0.5883

    print("\nFASE 1: Preparando dataset de desenvolvimento...")
    try:
        df_clean_dev = load_and_clean_data(PATH_RAW, is_test_set=False)
        df_dev_with_features = create_advanced_features(df_clean_dev)
        print(f"Dados carregados e processados")
    except Exception as e:
        print(f"Erro no carregamento: {e}")
        return

    y = df_dev_with_features['INADIMPLENTE']
    groups = df_dev_with_features['ID_CLIENTE']
    X_raw = df_dev_with_features.drop(columns=['INADIMPLENTE'])

    '''Encoding categórico'''
    categorical_cols = X_raw.select_dtypes(
        include=['object', 'category']).columns
    X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)


    cols_to_drop = ['ID_CLIENTE', 'DATA_EMISSAO_DOCUMENTO', 'DATA_PAGAMENTO',
                    'DATA_VENCIMENTO', 'DATA_CADASTRO', 'SAFRA_REF']
    X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])
    X.fillna(0, inplace=True)

    print(f"Dataset preparado: {X.shape[0]} amostras, {X.shape[1]} features")

    '''FASE 2: VALIDAÇÃO CRUZADA'''
    print("\nFASE 2: Validando modelo com threshold otimizado...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]

    auc_scores, recall_scores, precision_scores, f1_scores = [], [], [], []

    for fold, (train_index, val_index) in enumerate(sgkf.split(X, y, groups=groups)):
        print(f"  Processando fold {fold+1}/5...")

        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            random_state=42,
            **best_params
        )
        model.fit(X_train, y_train)

        '''Predições com threshold otimizado'''
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred_class = (y_pred_proba >= OPTIMAL_THRESHOLD).astype(int)

        '''Métricas'''
        auc_scores.append(roc_auc_score(y_val, y_pred_proba))
        recall_scores.append(recall_score(y_val, y_pred_class))
        precision_scores.append(precision_score(
            y_val, y_pred_class, zero_division=0))
        f1_scores.append(f1_score(y_val, y_pred_class))

    print("\nRESULTADOS DA VALIDAÇÃO CRUZADA:")
    print(
        f"  -  AUC Médio:       {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")
    print(
        f"  -  Recall Médio:    {np.mean(recall_scores):.4f} (+/- {np.std(recall_scores):.4f})")
    print(
        f"  -  Precision Médio: {np.mean(precision_scores):.4f} (+/- {np.std(precision_scores):.4f})")
    print(
        f"  -  F1-Score Médio:  {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")

    '''FASE 3: MODELO FINAL E SUBMISSÃO'''
    print("\n FASE 3: Treinando modelo final e gerando submissão...")

    xgb_final_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': scale_pos_weight,
        'use_label_encoder': False,
        'random_state': 42,
        **best_params
    }
    final_model = train_final_model(X, y, xgb.XGBClassifier, xgb_final_params)

    try:
        df_clean_test = load_and_clean_data(PATH_RAW, is_test_set=True)
        print(f" Registros carregados: {len(df_clean_test)}")

        df_teste_with_features = create_advanced_features(
            df_clean_test, is_test_set=True)
        print(f" Registros após features: {len(df_teste_with_features)}")

        submission_ids = df_teste_with_features[[
            'ID_CLIENTE', 'SAFRA_REF', 'DATA_EMISSAO_DOCUMENTO', 'DATA_VENCIMENTO']].copy()

        X_teste_raw = df_teste_with_features
        categorical_cols_test = X_teste_raw.select_dtypes(
            include=['object', 'category']).columns
        X_teste_processed = pd.get_dummies(
            X_teste_raw, columns=categorical_cols_test, drop_first=True)
        X_teste_processed = X_teste_processed.drop(
            columns=[col for col in cols_to_drop if col in X_teste_processed.columns])
        X_teste = X_teste_processed.reindex(columns=X.columns, fill_value=0)

        print(f"Base de teste preparada: {X_teste.shape[0]} registros")

        final_probabilities = final_model.predict_proba(X_teste)[:, 1]

        submission_df = submission_ids.copy()
        submission_df['PROBABILIDADE_INADIMPLENCIA'] = final_probabilities
        submission_df.to_csv(f'{path_processed}/submissao_case.csv', index=False, decimal=',')

        print(f"\n\o/ PIPELINE CONCLUÍDO COM SUCESSO!")
        print(
            f"Arquivo 'submissao_case.csv' gerado com {len(submission_df)} predições")
        print(
            f"Range de probabilidades: {final_probabilities.min():.4f} - {final_probabilities.max():.4f}")

        return submission_df

    except Exception as e:
        print(f"Erro no processamento de teste: {e}")
        return None


if __name__ == "__main__":
    try:
        result = main()
        if result is not None:
            print("\nExecução finalizada sem erros!")
        else:
            print("\nExecução finalizada com erros")
    except Exception as e:
        print(f"\nErro durante execução: {e}")
        import traceback
        traceback.print_exc()
