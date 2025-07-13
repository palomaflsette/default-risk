import pandas as pd
import numpy as np


def prepare_data_for_modeling(path_to_raw_data: str) -> pd.DataFrame:
    """
    Executa o pipeline completo de carregamento, limpeza, junção e 
    engenharia de features para preparar os dados para a modelagem.

    Args:
        path_to_raw_data (str): O caminho para a pasta contendo os arquivos .csv brutos.

    Returns:
        pd.DataFrame: Um DataFrame limpo, enriquecido e pronto para a modelagem.
    """
    print("Iniciando o pipeline de preparação de dados...")

    # --- 1. Carregamento ---
    try:
        base_pagamentos = pd.read_csv(
            f'{path_to_raw_data}/base_pagamentos_desenvolvimento.csv', delimiter=';')
        base_cadastral = pd.read_csv(
            f'{path_to_raw_data}/base_cadastral.csv', delimiter=';')
        base_info = pd.read_csv(
            f'{path_to_raw_data}/base_info.csv', delimiter=';')
    except FileNotFoundError as e:
        print(f"Erro ao carregar arquivos: {e}")
        return None
    print("Dados carregados.")

    # --- 2. Limpeza e Conversão de Tipos ---
    for df, cols in [(base_pagamentos, ['DATA_PAGAMENTO', 'DATA_VENCIMENTO', 'DATA_EMISSAO_DOCUMENTO', 'SAFRA_REF']),
                     (base_cadastral, ['DATA_CADASTRO']),
                     (base_info, ['SAFRA_REF'])]:
        for col in cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    base_pagamentos.dropna(subset=[
                           'DATA_PAGAMENTO', 'DATA_VENCIMENTO', 'DATA_EMISSAO_DOCUMENTO'], inplace=True)
    print("Limpeza e conversão de tipos concluída.")

    # --- 3. Criação da Variável-Alvo ---
    dias_de_atraso = (
        base_pagamentos['DATA_PAGAMENTO'] - base_pagamentos['DATA_VENCIMENTO']).dt.days
    base_pagamentos['INADIMPLENTE'] = np.where(
        dias_de_atraso >= 5, 1, 0).astype(int)
    print("Variável-alvo 'INADIMPLENTE' criada.")

    # --- 4. Junção das Bases (Merges) ---
    df_merged = pd.merge(base_pagamentos, base_cadastral,
                         on='ID_CLIENTE', how='left')
    df_final = pd.merge(df_merged, base_info, on=[
                        'ID_CLIENTE', 'SAFRA_REF'], how='left')
    print("Junção das bases de dados concluída.")

    # --- 5. Engenharia de Features ---
    # Temporais
    df_final['PRAZO_PAGAMENTO_DIAS'] = (
        df_final['DATA_VENCIMENTO'] - df_final['DATA_EMISSAO_DOCUMENTO']).dt.days
    df_final['IDADE_CLIENTE_NA_TRANSACAO'] = (
        df_final['DATA_EMISSAO_DOCUMENTO'] - df_final['DATA_CADASTRO']).dt.days

    # Financeiras e Operacionais
    epsilon = 1e-6
    df_final['ALAVANCAGEM_FINANCEIRA'] = df_final['VALOR_A_PAGAR'] / \
        (df_final['RENDA_MES_ANTERIOR'] + epsilon)
    df_final['PESO_EMPRESTIMO_POR_FUNCIONARIO'] = df_final['VALOR_A_PAGAR'] / \
        (df_final['NO_FUNCIONARIOS'] + epsilon)

    # Limpeza pós-feature engineering
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.dropna(subset=['IDADE_CLIENTE_NA_TRANSACAO'], inplace=True)
    df_final = df_final[df_final['IDADE_CLIENTE_NA_TRANSACAO'] >= 0].copy()
    print("Engenharia de features concluída.")

    # --- 6. Preparação Final para Modelagem ---
    cols_to_drop = ['ID_CLIENTE', 'SAFRA_REF', 'DATA_EMISSAO_DOCUMENTO', 'DATA_PAGAMENTO',
                    'DATA_VENCIMENTO', 'DATA_CADASTRO', 'FLAG_PF', 'CEP_2_DIG', 'DOMINIO_EMAIL', 'DDD']
    df_model = df_final.drop(columns=cols_to_drop)

    # Tratamento de nulos restantes
    df_model.fillna(0, inplace=True)

    # One-Hot Encoding para variáveis categóricas
    categorical_cols = df_model.select_dtypes(include=['object']).columns
    df_model_encoded = pd.get_dummies(
        df_model, columns=categorical_cols, drop_first=True)
    print("Preparação final para modelagem concluída.")

    return df_model_encoded
