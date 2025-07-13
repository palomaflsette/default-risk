import pandas as pd
import numpy as np


def load_and_clean_data(path_to_raw_data: str) -> pd.DataFrame:
    """
    Carrega os dados brutos, realiza a limpeza inicial, conversões de tipo,
    cria a variável-alvo e une as bases.

    Etapas:
        1. Carregamento;
        2. Conversão de Tipos e limpeza
        3. Criação da variável-alvo
        4. junções
    
    Args:
        path_to_raw_data (str): O caminho para a pasta com os arquivos brutos.

    Returns:
        pd.DataFrame: Um DataFrame limpo e unido, pronto para a engenharia de features.
    """
    print("Iniciando pipeline de preparação de dados...")
    
    '''1. Carregamento'''
    base_pagamentos = pd.read_csv(f'{path_to_raw_data}/base_pagamentos_desenvolvimento.csv', delimiter=';')
    base_cadastral = pd.read_csv(f'{path_to_raw_data}/base_cadastral.csv', delimiter=';')
    base_info = pd.read_csv(f'{path_to_raw_data}/base_info.csv', delimiter=';')
    print("Dados brutos carregados.")

    '''2. Conversão de Tipos e Limpeza'''
    for df, cols in [(base_pagamentos, ['DATA_PAGAMENTO', 'DATA_VENCIMENTO', 'DATA_EMISSAO_DOCUMENTO', 'SAFRA_REF']),
                     (base_cadastral, ['DATA_CADASTRO']),
                     (base_info, ['SAFRA_REF'])]:
        for col in cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    base_pagamentos.dropna(subset=['DATA_PAGAMENTO', 'DATA_VENCIMENTO', 'DATA_EMISSAO_DOCUMENTO'], inplace=True)

    '''3. Criação da Variável-Alvo'''
    dias_de_atraso = (base_pagamentos['DATA_PAGAMENTO'] - base_pagamentos['DATA_VENCIMENTO']).dt.days
    base_pagamentos['INADIMPLENTE'] = np.where(dias_de_atraso >= 5, 1, 0)

    '''4. Junções'''
    df_merged = pd.merge(base_pagamentos, base_cadastral, on='ID_CLIENTE', how='left')
    df_clean = pd.merge(df_merged, base_info, on=['ID_CLIENTE', 'SAFRA_REF'], how='left')

    
    print("Processamento de dados concluído.")
    return df_clean