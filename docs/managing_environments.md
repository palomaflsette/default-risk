# Guia de Gerenciamento de Ambiente e Dependências

Este documento descreve o fluxo de trabalho para gerenciar o ambiente de desenvolvimento deste projeto. O objetivo é garantir um ambiente **reprodutível**, **estável** e **consistente** para todos os colaboradores.

A nossa filosofia é usar o `conda` como o gerenciador principal de ambiente e pacotes, e o arquivo `environment.yml` como a **única fonte da verdade** para declarar as dependências do projeto.

---

## 1. Configuração Inicial do Ambiente

Para configurar o ambiente do projeto pela primeira vez, siga os passos abaixo.

1. **Clone o repositório** (se aplicável).
2. **Crie o ambiente Conda** a partir do arquivo `environment.yml`. Este comando irá instalar todas as dependências listadas, incluindo o Python na versão especificada.

   ```bash
   conda env create -f environment.yml
   ```
3. **Ative o ambiente recém-criado**. Você precisará fazer isso toda vez que for trabalhar no projeto.

   ```bash
   conda activate datarisk_env
   ```

---

## 2. O Fluxo de Trabalho para Adicionar uma Nova Dependência

Quando for necessário adicionar uma nova biblioteca ao projeto, **nunca** use `conda install <pacote>` ou `pip install <pacote>` diretamente no terminal. Siga este fluxo de trabalho deliberado:

### Passo 1: Pesquisar o Pacote

Primeiro, verifique se o pacote está disponível no canal `conda-forge`, que é o nosso preferido.

```bash
conda search -c conda-forge <nome-do-pacote>
```

### Passo 2: Declarar a Dependência no `environment.yml`

Este é o passo mais importante. Edite o arquivo `environment.yml` para adicionar a nova dependência.

* **Se o pacote foi encontrado no Conda**, adicione-o à lista principal de `dependencies`:

  ```yaml
  dependencies:
    - pandas
    - scikit-learn
    - xgboost # <-- Nova dependência adicionada
  ```
* **Se o pacote NÃO foi encontrado no Conda**, adicione-o na sub-seção `pip`. Certifique-se de que `- pip` já está na lista principal.

  ```yaml
  dependencies:
    - pandas
    - pip # <-- Garanta que esta linha existe
    - pip:
      - alguma-lib-especifica==1.2.3 # <-- Nova dependência via Pip
  ```

### Passo 3: Sincronizar o Ambiente

Com o arquivo `environment.yml` salvo, execute o seguinte comando para que o `conda` atualize seu ambiente para corresponder ao que foi declarado.

```bash
conda env update --file env.yml --prune
```

* **`conda env update`**: Lê o arquivo e de forma inteligente adiciona, remove ou atualiza os pacotes.
* **`--prune`**: Remove quaisquer pacotes que foram instalados anteriormente no ambiente mas que não estão mais declarados no arquivo `.yml`, mantendo o ambiente limpo.

---

## 3. Regra de Ouro e Comandos a Evitar

* **REGRA DE OURO:** A única coisa que você edita manualmente é o **arquivo `environment.yml`**. O único comando que você usa para modificar o ambiente é **`conda env update`**.
* **COMANDOS A EVITAR:** No contexto deste fluxo de trabalho, evite os seguintes comandos, pois eles criam uma inconsistência entre o seu ambiente ativo e a sua declaração de dependências ("deriva de configuração"):

  * `conda install <pacote>`
  * `conda remove <pacote>`
  * `pip install <pacote>`
  * `pip uninstall <pacote>`

---

## 4. Gerando um "Lock File" (`requirements.txt`)

O `environment.yml` é a nossa "receita". Para garantir 100% de reprodutibilidade, podemos gerar um "lock file" (`requirements.txt`), que é como uma "fotografia" do ambiente com as versões exatas de *todas* as sub-dependências.

1. **Ative o ambiente** (`conda activate datarisk_env`).
2. **Gere o arquivo** com o `pip freeze`:

   ```bash
   pip freeze > requirements.txt
   ```

Este arquivo deve ser atualizado periodicamente e commitado no Git, mas lembre-se: a fonte da verdade para o desenvolvimento continua sendo o `environment.yml`.
