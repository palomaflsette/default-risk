## Guia de Execução

### Pré-requisitos e Configuração do Ambiente

Antes de executar qualquer código, é **essencial** configurar adequadamente o ambiente virtual. O projeto foi desenvolvido em **Python 3.10** e possui dependências específicas que devem ser instaladas em um ambiente isolado.

**Consulte o arquivo `docs/managing_environments.md` para instruções detalhadas sobre:**

- Criação e ativação de ambientes virtuais
- Instalação de dependências via pip
- Boas práticas de gerenciamento de dependências
- Resolução de problemas comuns

### Dependências Principais

O projeto utiliza as seguintes bibliotecas principais:

```
pandas>=1.5.0          # Manipulação de dados
numpy>=1.21.0          # Computação numérica
scikit-learn>=1.2.0    # Machine learning
xgboost>=1.7.0         # Gradient boosting
matplotlib>=3.5.0      # Visualizações básicas
seaborn>=0.11.0        # Visualizações estatísticas
openpyxl>=3.0.0        # Manipulação de arquivos Excel
```

### Ordem Recomendada de Execução

#### 1. Análise Exploratória e Engenharia de Features

**Arquivo: `notebooks/1.0-EDA_e_Feature_Engineering.ipynb`**

Este notebook representa o **núcleo conceitual** do projeto e deve ser executado **PREFERENCIALMENTE em primeiro lugar**. Ele contém:

- **Formulação de Hipóteses de Negócio**: Desenvolvimento de hipóteses fundamentadas sobre fatores de risco
- **Análise Exploratória Sistemática**: Investigação estruturada em camadas (perfil estático, comportamento temporal, contexto financeiro)
- **Validação Visual de Hipóteses**: Comprovação ou refutação das hipóteses através de visualizações especializadas
- **Engenharia de Features Avançada**: Criação de variáveis preditivas complexas, incluindo:
  - Features de alavancagem financeira
  - Análise temporal com _lag features_ e _rolling aggregates_
  - Features de interação explícita
  - Discretização inteligente de variáveis contínuas
- **Análises Complementares**: Correlação, t-SNE e outros métodos exploratórios

**Por que começar aqui?**
Este notebook estabelece toda a **lógica de negócio** e **fundamentação teórica** que sustenta as decisões tomadas na modelagem. Sem compreender os _insights_ desenvolvidos nesta etapa, as escolhas técnicas do notebook seguinte podem parecer arbitrárias.

#### 2. Modelagem e Avaliação

**Arquivo: `notebooks/2.0-Modelagem_e_Avaliacao.ipynb`**

Este notebook implementa a solução preditiva baseada nos _insights_ do notebook anterior:

- **Estratégia de Validação Rigorosa**: Uso de StratifiedGroupKFold para evitar data leakage
- **Comparação de Modelos**: Avaliação de Regressão Logística vs XGBoost
- **Otimização de Hiperparâmetros**: Busca sistemática pelos melhores parâmetros
- **Calibração de Threshold**: Otimização do ponto de corte para maximizar F1-Score
- **Validação Final**: Avaliação robusta com métricas de negócio (AUC, _Recall_, _Precision_)
- **Geração de Submissão**: Criação do arquivo final de predições

**Características técnicas importantes:**

- Tratamento adequado do desbalanceamento de classes
- Foco em métricas de negócio relevantes (priorização do _Recall_ equilibrado com _Precision_)
- Validação cruzada que respeita a estrutura temporal dos dados

#### 3. Pipeline Automatizado (Opcional)

**Arquivo: `run_pipeline.py`**

Este script oferece uma execução automatizada e simplificada do processo completo:

```bash
python run_pipeline.py
```

Quando usar:

- Após compreender completamente os notebooks
- Para simulação de um ambiente de produção
- Para reprodução rápida dos resultados finais

> **Importante:** O pipeline automatizado representa uma **ponte conceitual entre desenvolvimento e produção**, introduzindo os fundamentos de MLOps de forma didática. Enquanto os notebooks privilegiam a exploração e compreensão, este script demonstra como o conhecimento adquirido seria estruturado em um **sistema automatizado e versionado**. É o primeiro degrau na escada evolutiva que leva a arquiteturas mais robustas: orquestração com Apache Airflow, containerização com Docker, monitoramento contínuo e deploy automatizado. Pense nele como um "proof-of-concept" que prepara o terreno para implementações _enterprise_ de MLOps.

#### 4. Verificação de Qualidade da Submissão (Recomendado)

**Arquivo: `verificar_submissao.py`**

Este script utilitário oferece uma análise detalhada da qualidade e consistência do arquivo de submissão gerado:

```bash
python verificar_submissao.py
```

**Exemplo de saída:**

```bash
COBERTURA DE REGISTROS:
   Base teste original: 12,275
   Submissão gerada:    12,237
   Diferença:           38 (+0.3%)

CLIENTES ÚNICOS:
   Base teste:   976 clientes
   Submissão:    955 clientes

SAFRAS ÚNICAS:
   Base teste:   5
   Submissão:    5

ANÁLISE DAS PROBABILIDADES:
   Mínima:     0.2041
   Máxima:     0.8948
   Média:      0.3761
   Mediana:    0.3388
   Desvio:     0.1272

VERIFICAÇÕES DE QUALIDADE:
   Valores nulos:           0
   Probabilidades inválidas: 0
   Transações por chave:     3,410 chaves únicas
   Múltiplas transações:     8,827 casos

ESTRUTURA DO ARQUIVO:
   Colunas esperadas: ID_CLIENTE, SAFRA_REF, PROBABILIDADE_INADIMPLENCIA
   Colunas presentes: ID_CLIENTE, SAFRA_REF, PROBABILIDADE_INADIMPLENCIA
   Estrutura correta: SIM

DISTRIBUIÇÃO DE RISCO:
   Baixo risco (≤0.3):  3,171 (25.9%)
   Médio risco (0.3-0.7): 8,419 (68.8%)
   Alto risco (>0.7):    647 (5.3%)

AMOSTRA DA SUBMISSÃO:
      ID_CLIENTE  SAFRA_REF  PROBABILIDADE_INADIMPLENCIA
8784237149961904 2021-07-01                     0.283395
8784237149961904 2021-07-01                     0.301155
8784237149961904 2021-07-01                     0.287099
8784237149961904 2021-08-01                     0.266573
8784237149961904 2021-08-01                     0.294347
8784237149961904 2021-08-01                     0.356639
8784237149961904 2021-08-01                     0.358506
8784237149961904 2021-08-01                     0.344710
8784237149961904 2021-08-01                     0.358537
8784237149961904 2021-08-01                     0.359535

============================================================
Todos os critérios de qualidade foram atendidos.
Arquivo pronto para submissão!
============================================================
```


### Principais Descobertas e Contribuições

#### _Insights_ de Negócio Identificados

1. Alavancagem Financeira como Preditor Principal: A relação entre valor da dívida e renda mensal revelou-se o indicador de risco mais poderoso
2. Importância do Histórico de Relacionamento: Clientes mais novos apresentam risco significativamente maior de inadimplência
3. Efeito da Completude Cadastral: A ausência de informações cadastrais (campos "NÃO INFORMADO") emergiu como um dos sinais de alerta mais fortes
4. Interações Complexas: O risco varia significativamente dependendo da combinação entre porte da empresa e segmento industrial

#### Inovações Técnicas Implementadas

1. Engenharia de Features Temporal: Criação de lag features e rolling aggregates que capturam a dinâmica comportamental dos clientes
2. Features de Interação Explícita: Transformação de insights visuais em variáveis categóricas que facilitam o aprendizado do modelo
3. Validação Temporal Rigorosa: Implementação de StratifiedGroupKFold garantindo que transações do mesmo cliente não vazem entre treino e validação
4. Otimização de Threshold Baseada em Negócio: Calibração do ponto de corte focada na maximização do valor de negócio, não apenas na acurácia

#### Métricas de Performance Final
O modelo final (XGBoost otimizado) apresentou as seguintes métricas na validação cruzada:

- AUC-ROC: 0.7298 (capacidade de ranking/discriminação)
- _Recall_: 0.4341 (captura 43% dos inadimplentes reais)
- _Precision_: 0.3027 (30% dos alertas são verdadeiros)
- F1-_Score_: 0.3541 (equilíbrio otimizado entre _precision_ e _recall_)

Estas métricas representam um desempenho sólido para um problema de classes desbalanceadas (~ 7.0% de inadimplentes), oferecendo valor significativo para ações proativas de cobrança.


### Considerações sobre Reprodutibilidade

- Todos os processos utilizam `random_state=42` para garantir reprodutibilidade
- As funções modulares em `src/` permitem reutilização em diferentes contextos
- O pipeline automatizado simula um ambiente de produção real
- Documentação detalhada facilita compreensão e manutenção

### Estrutura dos Dados
O projeto trabalha com quatro bases de dados principais:

- `base_pagamentos_desenvolvimento.csv`: Histórico de transações com target
- `base_pagamentos_teste.csv`: Transações para predição
- `base_cadastral.csv`: Informações estáticas dos clientes
- `base_info.csv`: Dados mensais de acompanhamento

Dísponíveis em `data/raw`. Consulte os notebooks para descrição detalhada do _schema_ e relacionamentos.

## Considerações Finais

Este projeto demonstra uma abordagem científica rigorosa para problemas de _Data Science_, priorizando o entendimento antes da automação. A metodologia empregada pode ser adaptada para diversos contextos de análise de risco, servindo como referência para desenvolvimento de soluções de _machine learning_ em ambientes corporativos.

A combinação de análise exploratória aprofundada, engenharia de _features_ criativa e modelagem robusta resulta em uma solução que não apenas apresenta boa performance técnica, mas também oferece _insights_ acionáveis para o negócio.
