import pandas as pd
import numpy as np


def create_advanced_features(df: pd.DataFrame, training_columns: list = None, is_test_set: bool = False) -> pd.DataFrame:
    """
    Recebe um DataFrame limpo e aplica a engenharia de features avançada.
    """
    print("Iniciando pipeline de engenharia de features...")
    df_features = df.copy()

    df_features.sort_values(by=['ID_CLIENTE', 'SAFRA_REF'], inplace=True)

    '''Engenharia de Features'''
    grouped_by_cliente = df_features.groupby('ID_CLIENTE')
    df_features['RENDA_LAG1'] = grouped_by_cliente['RENDA_MES_ANTERIOR'].shift(
        1)
    df_features['RENDA_MEDIA_3M'] = grouped_by_cliente['RENDA_MES_ANTERIOR'].rolling(
        window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    df_features['RENDA_STD_3M'] = grouped_by_cliente['RENDA_MES_ANTERIOR'].rolling(
        window=3, min_periods=1).std().reset_index(level=0, drop=True)

    '''Verificar se as colunas de data existem antes de calcular'''
    if 'DATA_EMISSAO_DOCUMENTO' in df_features.columns and 'DATA_CADASTRO' in df_features.columns:
        df_features['IDADE_CLIENTE_NA_TRANSACAO'] = (
            df_features['DATA_EMISSAO_DOCUMENTO'] - df_features['DATA_CADASTRO']).dt.days

    epsilon = 1e-6
    df_features['ALAVANCAGEM_FINANCEIRA'] = df_features['VALOR_A_PAGAR'] / \
        (df_features['RENDA_MES_ANTERIOR'] + epsilon)
    df_features['PESO_EMPRESTIMO_POR_FUNCIONARIO'] = df_features['VALOR_A_PAGAR'] / \
        (df_features['NO_FUNCIONARIOS'] + epsilon)

    '''Tratamento de colunas categóricas'''
    if 'PORTE' in df_features.columns:
        df_features['PORTE'].fillna('NÃO_INFORMADO', inplace=True)
    if 'SEGMENTO_INDUSTRIAL' in df_features.columns:
        df_features['SEGMENTO_INDUSTRIAL'].fillna(
            'NÃO_INFORMADO', inplace=True)
        df_features['PERFIL_EMPRESA'] = df_features['PORTE'] + \
            '_' + df_features['SEGMENTO_INDUSTRIAL']

    '''Tratamento da idade do cliente'''
    if 'IDADE_CLIENTE_NA_TRANSACAO' in df_features.columns:
        df_features.dropna(subset=['IDADE_CLIENTE_NA_TRANSACAO'], inplace=True)
        df_features = df_features[df_features['IDADE_CLIENTE_NA_TRANSACAO'] >= 0].copy(
        )
        bins = [-1, 180, 730, np.inf]
        labels = ['1_Novissimo (0-6m)', '2_Recente (6m-2a)',
                  '3_Estabelecido (2a+)']
        df_features['FAIXA_IDADE_CLIENTE'] = pd.cut(
            df_features['IDADE_CLIENTE_NA_TRANSACAO'], bins=bins, labels=labels)

    '''PREPARAÇÃO PARA MODELAGEM''' 
    
    if is_test_set:
        cols_to_drop_ideal = ['DATA_CADASTRO', 'FLAG_PF', 'CEP_2_DIG',
                              'DOMINIO_EMAIL', 'DDD', 'PORTE', 'SEGMENTO_INDUSTRIAL']
    else:
        cols_to_drop_ideal = ['SAFRA_REF', 'DATA_EMISSAO_DOCUMENTO', 'DATA_PAGAMENTO', 'DATA_VENCIMENTO',
                              'DATA_CADASTRO', 'FLAG_PF', 'CEP_2_DIG', 'DOMINIO_EMAIL', 'DDD', 'PORTE',
                              'SEGMENTO_INDUSTRIAL']

    '''Removendo apenas as colunas que existem'''
    cols_to_drop_existing = [
        col for col in cols_to_drop_ideal if col in df_features.columns]
    df_model = df_features.drop(columns=cols_to_drop_existing)

    '''TRATANDO DE VALORES INFINITOS E NULOS'''

    df_model.replace([np.inf, -np.inf], 0, inplace=True)

    numeric_cols = df_model.select_dtypes(include=[np.number]).columns
    if 'ID_CLIENTE' in numeric_cols:
        numeric_cols = numeric_cols.drop('ID_CLIENTE')

    for col in numeric_cols:
        df_model[col].fillna(0, inplace=True)

    '''Colunas categóricas (object e category)'''
    categorical_cols = df_model.select_dtypes(
        include=['object', 'category']).columns

    for col in categorical_cols:
        if pd.api.types.is_categorical_dtype(df_model[col]):
            if 'DESCONHECIDO' not in df_model[col].cat.categories:
                df_model[col] = df_model[col].cat.add_categories(
                    ['DESCONHECIDO'])
            df_model[col].fillna('DESCONHECIDO', inplace=True)
        else:
            df_model[col].fillna('DESCONHECIDO', inplace=True)

    # One-Hot Encoding
    categorical_cols = df_model.select_dtypes(
        include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        df_model_encoded = pd.get_dummies(
            df_model, columns=categorical_cols, drop_first=True)
    else:
        df_model_encoded = df_model.copy()

    print("Engenharia de features concluída.")
    return df_model_encoded
