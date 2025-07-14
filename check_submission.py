"""
Script para verificação da qualidade da submissão do case de risco de crédito.

Uso:
    python.exe -u verificar_submissao.py
    
Ou com arquivos customizados:
    python.exe -u verificar_submissao.py --teste data/raw/base_pagamentos_teste.csv --submissao data/processed/submissao_case.csv
"""

import pandas as pd
import argparse
import os
from pathlib import Path


def verificar_qualidade_submissao(caminho_teste, caminho_submissao):
    """
    Verifica a qualidade e consistência da submissão gerada.
    
    Args:
        caminho_teste (str): Caminho para base de teste original
        caminho_submissao (str): Caminho para arquivo de submissão
    """

    print("VERIFICAÇÃO DE QUALIDADE DA SUBMISSÃO")
    print("=" * 60)

    try:
        print("Carregando arquivos...")
        base_teste = pd.read_csv(caminho_teste, delimiter=';')
        submissao = pd.read_csv(caminho_submissao)

        if submissao['PROBABILIDADE_INADIMPLENCIA'].dtype == 'object':
            submissao['PROBABILIDADE_INADIMPLENCIA'] = submissao['PROBABILIDADE_INADIMPLENCIA'].str.replace(
                ',', '.').astype(float)

        print("Arquivos carregados com sucesso!\n")

        print(f"COBERTURA DE REGISTROS:")
        print(f"   Base teste original: {len(base_teste):,}")
        print(f"   Submissão gerada:    {len(submissao):,}")
        diferenca = len(base_teste) - len(submissao)
        percentual = (diferenca / len(base_teste)) * 100
        print(f"   Diferença:           {diferenca:,} ({percentual:+.1f}%)")

        print(f"\nCLIENTES ÚNICOS:")
        print(
            f"   Base teste:   {base_teste['ID_CLIENTE'].nunique():,} clientes")
        print(
            f"   Submissão:    {submissao['ID_CLIENTE'].nunique():,} clientes")

        print(f"\nSAFRAS ÚNICAS:")
        print(f"   Base teste:   {base_teste['SAFRA_REF'].nunique()}")
        print(f"   Submissão:    {submissao['SAFRA_REF'].nunique()}")

        probs = submissao['PROBABILIDADE_INADIMPLENCIA']
        print(f"\nANÁLISE DAS PROBABILIDADES:")
        print(f"   Mínima:     {probs.min():.4f}")
        print(f"   Máxima:     {probs.max():.4f}")
        print(f"   Média:      {probs.mean():.4f}")
        print(f"   Mediana:    {probs.median():.4f}")
        print(f"   Desvio:     {probs.std():.4f}")

        print(f"\nVERIFICAÇÕES DE QUALIDADE:")
        valores_nulos = submissao.isnull().sum().sum()
        print(f"   Valores nulos:           {valores_nulos}")

        probs_invalidas = ((probs < 0) | (probs > 1)).sum()
        print(f"   Probabilidades inválidas: {probs_invalidas}")

        duplicados = submissao.duplicated(['ID_CLIENTE', 'SAFRA_REF']).sum()
        print(
            f"   Transações por chave:     {len(submissao) - duplicados:,} chaves únicas")
        print(f"   Múltiplas transações:     {duplicados:,} casos")

        print(f"\nESTRUTURA DO ARQUIVO:")
        print(f"   Colunas esperadas: ID_CLIENTE, SAFRA_REF, PROBABILIDADE_INADIMPLENCIA")
        print(f"   Colunas presentes: {', '.join(submissao.columns)}")

        colunas_corretas = set(submissao.columns) == {
            'ID_CLIENTE', 'SAFRA_REF', 'PROBABILIDADE_INADIMPLENCIA'}
        print(
            f"   Estrutura correta: {'SIM' if colunas_corretas else '❌ NÃO'}")

        print(f"\nDISTRIBUIÇÃO DE RISCO:")
        baixo_risco = (probs <= 0.3).sum()
        medio_risco = ((probs > 0.3) & (probs <= 0.7)).sum()
        alto_risco = (probs > 0.7).sum()

        print(
            f"   Baixo risco (≤0.3):  {baixo_risco:,} ({baixo_risco/len(submissao)*100:.1f}%)")
        print(
            f"   Médio risco (0.3-0.7): {medio_risco:,} ({medio_risco/len(submissao)*100:.1f}%)")
        print(
            f"   Alto risco (>0.7):    {alto_risco:,} ({alto_risco/len(submissao)*100:.1f}%)")

        print(f"\nAMOSTRA DA SUBMISSÃO:")
        print(submissao.head(10).to_string(index=False))

        print(f"\n" + "=" * 60)

        cobertura_ok = percentual < 5.0  # Menos de 5% de perda
        estrutura_ok = colunas_corretas
        probabilidades_ok = (valores_nulos == 0) and (probs_invalidas == 0)
        distribuicao_ok = (probs.min() > 0.1) and (
            probs.max() < 0.95)  # Range razoável

        if all([cobertura_ok, estrutura_ok, probabilidades_ok, distribuicao_ok]):
            print("Todos os critérios de qualidade foram atendidos.")
            print("Arquivo pronto para submissão!")
        elif cobertura_ok and estrutura_ok and probabilidades_ok:
            print("Critérios essenciais atendidos.")
            print("Pequenos ajustes podem ser feitos, mas está aprovada.")
        else:
            print("VEREDICTO: SUBMISSÃO PRECISA DE REVISÃO!")
            if not cobertura_ok:
                print("Cobertura baixa (>5% de registros perdidos)")
            if not estrutura_ok:
                print("Estrutura de colunas incorreta")
            if not probabilidades_ok:
                print("Problemas nas probabilidades")

        print("=" * 60)

    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado - {e}")
    except Exception as e:
        print(f"Erro durante verificação: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Verificar qualidade da submissão')
    parser.add_argument('--teste',
                        default='data/raw/base_pagamentos_teste.csv',
                        help='Caminho para base de teste original')
    parser.add_argument('--submissao',
                        default='data/processed/submissao_case.csv',
                        help='Caminho para arquivo de submissão')

    args = parser.parse_args()

    if not os.path.exists(args.teste):
        print(f"Arquivo de teste não encontrado: {args.teste}")
        return

    if not os.path.exists(args.submissao):
        print(f"Arquivo de submissão não encontrado: {args.submissao}")
        return

    verificar_qualidade_submissao(args.teste, args.submissao)


if __name__ == "__main__":
    main()
