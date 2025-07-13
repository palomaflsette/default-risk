import pandas as pd
import numpy as np


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe um DataFrame limpo e aplica a engenharia de features avançada.

    Args:
        df (pd.DataFrame): O DataFrame limpo da etapa de processamento.

    Returns:
        pd.DataFrame: O DataFrame final, enriquecido e pronto para modelagem.
    """
    print("Iniciando pipeline de engenharia de features...")
    df_features = df.copy()

    '''Ordenação para cálculos temporais'''
    df_features.sort_values(by=['ID_CLIENTE', 'SAFRA_REF'], inplace=True)

    '''5.1 Comportamento (Lags e Rolling)'''
    grouped_by_cliente = df_features.groupby('ID_CLIENTE')
    df_features['RENDA_LAG1'] = grouped_by_cliente['RENDA_MES_ANTERIOR'].shift(
        1)
    df_features['RENDA_MEDIA_3M'] = grouped_by_cliente['RENDA_MES_ANTERIOR'].rolling(
        window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    df_features['RENDA_STD_3M'] = grouped_by_cliente['RENDA_MES_ANTERIOR'].rolling(
        window=3, min_periods=1).std().reset_index(level=0, drop=True)

    '''5.2 Outras Features'''
    df_features['IDADE_CLIENTE_NA_TRANSACAO'] = (
        df_features['DATA_EMISSAO_DOCUMENTO'] - df_features['DATA_CADASTRO']).dt.days
    epsilon = 1e-6
    df_features['ALAVANCAGEM_FINANCEIRA'] = df_features['VALOR_A_PAGAR'] / \
        (df_features['RENDA_MES_ANTERIOR'] + epsilon)
    df_features['PESO_EMPRESTIMO_POR_FUNCIONARIO'] = df_features['VALOR_A_PAGAR'] / \
        (df_features['NO_FUNCIONARIOS'] + epsilon)

    '''5.3 Interação'''
    df_features['PORTE'].fillna('NÃO_INFORMADO', inplace=True)
    df_features['SEGMENTO_INDUSTRIAL'].fillna('NÃO_INFORMADO', inplace=True)
    df_features['PERFIL_EMPRESA'] = df_features['PORTE'] + \
        '_' + df_features['SEGMENTO_INDUSTRIAL']

    '''5.4 Binning'''
    df_features.dropna(subset=['IDADE_CLIENTE_NA_TRANSACAO'], inplace=True)
    df_features = df_features[df_features['IDADE_CLIENTE_NA_TRANSACAO'] >= 0].copy(
    )
    bins = [-1, 180, 730, np.inf]
    labels = ['1_Novissimo (0-6m)', '2_Recente (6m-2a)',
              '3_Estabelecido (2a+)']
    df_features['FAIXA_IDADE_CLIENTE'] = pd.cut(
        df_features['IDADE_CLIENTE_NA_TRANSACAO'], bins=bins, labels=labels)

    '''6. Preparação Final'''
    cols_to_drop = ['ID_CLIENTE', 'SAFRA_REF', 'DATA_EMISSAO_DOCUMENTO', 'DATA_PAGAMENTO', 'DATA_VENCIMENTO',
                    'DATA_CADASTRO', 'FLAG_PF', 'CEP_2_DIG', 'DOMINIO_EMAIL', 'DDD', 'PORTE', 'SEGMENTO_INDUSTRIAL']
    df_model = df_features.drop(columns=cols_to_drop)
    df_model.replace([np.inf, -np.inf], 0, inplace=True)
    df_model.fillna(0, inplace=True)

    categorical_cols = df_model.select_dtypes(
        include=['object', 'category']).columns
    df_model_encoded = pd.get_dummies(
        df_model, columns=categorical_cols, drop_first=True)

    print("Engenharia de features concluída.")
    return df_model_encoded
