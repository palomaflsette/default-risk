import os
import pandas as pd
import numpy as np



def load_data_and_setup_env(project_root_path: str):
    """
    Carrega os datasets principais e garante que os diretórios necessários existam.

    Args:
        project_root_path (str): O caminho absoluto para a pasta raiz do projeto.

    Returns:
        tuple: Uma tupla contendo os três DataFrames carregados 
               (base_pagamentos_dev, base_cadastral, base_info).
               Retorna (None, None, None) se ocorrer um erro de arquivo não encontrado.
    """
    from config import (BASE_PAGAMENTOS_DESENVOLVIMENTO,
                        BASE_CADASTRAL,
                        BASE_INFO, ASSETS,
                        PATH_PROCESSED, PATH_RAW)
    path_raw = os.path.join(project_root_path, PATH_RAW)
    print(f"Caminho absoluto para os dados brutos: {path_raw}")
    
    try:
        path_pagamentos_dev = os.path.join(
            project_root_path, BASE_PAGAMENTOS_DESENVOLVIMENTO)
        path_cadastral = os.path.join(project_root_path, BASE_CADASTRAL)
        path_base_info = os.path.join(project_root_path, BASE_INFO)
        path_assets = os.path.join(project_root_path, ASSETS)
        path_processed = os.path.join(project_root_path, PATH_PROCESSED)

        print(f"Verificando diretório: {path_assets}")
        os.makedirs(path_assets, exist_ok=True)

        print(f"Verificando diretório: {path_processed}")
        os.makedirs(path_processed, exist_ok=True)

        print(f"\nTentando carregar dados de: {path_pagamentos_dev}")
        base_pagamentos_dev = pd.read_csv(path_pagamentos_dev, delimiter=';')
        base_cadastral = pd.read_csv(path_cadastral, delimiter=';')
        base_info = pd.read_csv(path_base_info, delimiter=';')

        print("\nDados carregados com sucesso!")

        return base_pagamentos_dev, path_cadastral, path_base_info, path_processed, base_cadastral, base_info, path_raw, path_assets

    except FileNotFoundError as e:
        print(f"\nERRO CRÍTICO: Arquivo não encontrado! Verifique o caminho em seu 'config.py'.")
        print(f"Detalhes: {e}")
        return None, None, None, None, None, None, None, None

    except Exception as e:
        print(f"\nOcorreu um erro inesperado durante o carregamento: {e}")
        return None, None, None, None, None, None, None, None
    
    
def load_and_clean_data(path_to_raw_data: str, is_test_set: bool = False) -> pd.DataFrame:
    """
    Carrega os dados brutos, realiza a limpeza inicial, conversões de tipo,
    cria a variável-alvo (apenas para o dataset de treino) e une as bases.

    Args:
        path_to_raw_data (str): O caminho para a pasta contendo os arquivos .csv brutos.
        is_test_set (bool): Flag para indicar se estamos carregando o conjunto de teste.

    Returns:
        pd.DataFrame: Um DataFrame limpo e unido, pronto para a engenharia de features.
    """
    print(
        f"Iniciando pipeline de processamento de dados para o conjunto de {'teste' if is_test_set else 'desenvolvimento'}...")


    if is_test_set:
        base_pagamentos = pd.read_csv(
            f'{path_to_raw_data}/base_pagamentos_teste.csv', delimiter=';')
    else:
        base_pagamentos = pd.read_csv(
            f'{path_to_raw_data}/base_pagamentos_desenvolvimento.csv', delimiter=';')

    base_cadastral = pd.read_csv(
        f'{path_to_raw_data}/base_cadastral.csv', delimiter=';')
    base_info = pd.read_csv(f'{path_to_raw_data}/base_info.csv', delimiter=';')

    date_cols_pagamentos = ['DATA_VENCIMENTO',
                            'DATA_EMISSAO_DOCUMENTO', 'SAFRA_REF']
    if not is_test_set:
        date_cols_pagamentos.append('DATA_PAGAMENTO')

    for col in date_cols_pagamentos:
        base_pagamentos[col] = pd.to_datetime(
            base_pagamentos[col], errors='coerce')

    base_cadastral['DATA_CADASTRO'] = pd.to_datetime(
        base_cadastral['DATA_CADASTRO'], errors='coerce')
    base_info['SAFRA_REF'] = pd.to_datetime(
        base_info['SAFRA_REF'], errors='coerce')

    if not is_test_set:
        base_pagamentos.dropna(subset=['DATA_PAGAMENTO'], inplace=True)
        dias_de_atraso = (
            base_pagamentos['DATA_PAGAMENTO'] - base_pagamentos['DATA_VENCIMENTO']).dt.days
        base_pagamentos['INADIMPLENTE'] = np.where(dias_de_atraso >= 5, 1, 0)

    df_merged = pd.merge(base_pagamentos, base_cadastral,
                         on='ID_CLIENTE', how='left')
    df_clean = pd.merge(df_merged, base_info, on=[
                        'ID_CLIENTE', 'SAFRA_REF'], how='left')

    print("Processamento de dados concluído.")
    return df_clean
