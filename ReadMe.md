# Dashboard EPA - Visualização de dados de poços

![status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)
<!-- ![streamlit](https://img.shields.io/badge/streamlit-1.52.2-red) -->
![license](https://img.shields.io/badge/licenca-all%20rights%20reserved-blue)

<<<<<<< HEAD
## Visão geral
Aplicação Streamlit para carregamento de planilhas Excel e criação de gráficos customizáveis a partir de dados de monitoramento. O app lê múltiplas abas, identifica tabelas validas, normaliza colunas e oferece modos de visualização temporal ou por poço, incluindo suporte especial a resultados laboratoriais com status (SECO, FASE LIVRE, etc).

## ✨ Funcionalidades principais
- Upload de Excel (.xlsx) com validação e limpeza automática das abas.
- Visualização de tabelas filtradas (remove linhas "Acumulado" e formata datas).
- Criação de gráficos com 1 ou 2 eixos Y, com seleção de parâmetros.
- Modo temporal (séries por poço) e modo por poço (agregação por categoria).
- Modo laboratorial com tratamento de status e timeline (heatmap).

## 🧭 Arquitetura e fluxo de dados
1) O usuário faz upload do Excel na página inicial (`app/app.py`).
2) O serviço de dataset processa o arquivo e retorna um dicionário de DataFrames.
3) O app guarda o dataset em `st.session_state` e mostra a tabela.
4) A página de gráficos (`app/pages/create_graph.py`) permite mapear colunas, filtrar por período/poços e gerar os gráficos com Plotly.

## 🗂️ Estrutura do projeto
- `app/app.py`: página inicial, upload e preview da tabela.
- `app/pages/create_graph.py`: tela de criação de gráficos e filtros.
- `app/services/dataset_service.py`: leitura e limpeza de planilhas.
- `app/data/transformer.py`: transformação de planilhas em formato laboratorial.
- `app/charts/builder.py`: construção dos gráficos Plotly.
=======
## Visao geral
Aplicacao Streamlit para carregamento de planilhas Excel e criacao de graficos customizaveis a partir de dados de monitoramento. O app le multiplas abas, identifica tabelas validas, normaliza colunas e oferece modos de visualizacao temporal ou por poco/ponto, com tratamento especial para NA/NO/FL e status qualitativos.

## Funcionalidades principais
- Upload de Excel (.xlsx) com validacao e limpeza automatica das abas.
- Criacao de graficos com 1 ou 2 eixos Y, com selecao de parametros.
- Visualizacao por ponto/poco com NA/NO em barras e volume bombeado em linha.
- Tratamento de valores qualitativos: "Nao medido", "nd", "Destruido", "Soterrado", etc.
- Fase livre qualitativa (Odor, Oleoso, Pelicula) com marcadores dedicados.

## Arquitetura e fluxo de dados
1) O usuario faz upload do Excel na pagina inicial (`app/dashboard.py`).
2) O servico de dataset processa o arquivo e retorna um dicionario de DataFrames.
3) O app guarda o dataset em `st.session_state` e mostra status, avisos e abas ignoradas.
4) A pagina de graficos (`app/pages/create_graph.py`) normaliza datasets, agrega por data e ponto e gera os graficos com Plotly.

## Estrutura do projeto
- `app/dashboard.py`: pagina inicial, upload e estado do dataset.
- `app/pages/create_graph.py`: tela de criacao de graficos e filtros.
- `app/services/dataset_service.py`: leitura e limpeza de planilhas.
- `app/charts/builder.py`: construcao dos graficos Plotly.
>>>>>>> origin/main
- `app/core/state.py`: controle de estado no Streamlit.
 - `app/services/date_num_prep.py`: utilitarios de data e numero PT-BR.

<<<<<<< HEAD
## 🧪 Detalhes do processamento de Excel
O serviço de dataset:
- Ignora abas claramente de gráfico (nome ou conteúdo) quando existe aba tabular correspondente.
- Tenta encontrar cabeçalho e conteúdo tabular mesmo em planilhas com layout irregular.
- Remove colunas "Unnamed:*" e linhas/colunas vazias.
- Normaliza nomes de colunas e converte datas automaticamente quando possivel.
- Converte números com vírgula e remove colunas de mês/ano redundantes quando há data completa.
- Para arquivos contendo "Historico" no nome, monta um dataset "master" a partir de abas predefinidas e transforma o formato para linhas por amostra.

## 🧫 Modo laboratorial (status)
Quando o dataset contem colunas no padrão `<param>__num` e `<param>__status`, a página de gráficos habilita um modo especial:
- "Valores dissolvidos": plota apenas resultados numéricos, com marcadores para SECO, FASE LIVRE e MISSING.
- "Timeline (status)": heatmap por poço x data com prioridade de status.

## ✅ Requisitos
=======
## Detalhes do processamento de Excel
O servico de dataset:
- Ignora abas claramente de grafico (nome ou conteudo) quando existe aba tabular correspondente.
- Tenta encontrar cabecalho e conteudo tabular mesmo em planilhas com layout irregular.
- Remove colunas "Unnamed:*" e linhas/colunas vazias.
- Normaliza nomes de colunas e converte datas automaticamente quando possivel.
- Converte numeros com virgula e remove colunas de mes/ano redundantes quando ha data completa.
- Interpreta NA/NO/FL com tratamento de numericos e qualitativos (inclusive "nm" como "Nao medido").

## Requisitos
>>>>>>> origin/main
- Python 3.12
- Dependências em `requirements.txt`

## Como rodar localmente
Crie e ative o ambiente virtual:
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

Instale as dependências:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Inicie o app:
```bash
streamlit run ./app/dashboard.py
```

O app fica disponível em `http://localhost:8501`.

## Como usar
1) Abra o app e envie um Excel (.xlsx) pela barra lateral.
2) Valide se as abas foram carregadas (avisos e abas ignoradas aparecem na tela).
<<<<<<< HEAD
3) Visualize os dados na tabela principal.
4) Clique em "Criar grafico" para acessar a página de gráficos.
5) Selecione colunas de data e poço/ponto, parâmetros e o tipo de gráfico.

## 💡 Observações e dicas
- Se uma aba não tiver tabela válida, ela será ignorada automaticamente.
- Para o modo temporal, a coluna de data precisa ser reconhecida ou selecionada manualmente.
- O modo por poço agrega valores (média, mediana, mínimo, máximo, soma).
=======
3) Clique em "Criar grafico" para acessar a pagina de graficos.
4) Selecione o tipo de visualizacao, os pocos e parametros.

## Fluxo de desenvolvimento
1) Crie uma branch propria de desenvolvimento.
2) Dê pull da `main`, desenvolva as mudancas e faça push da sua branch.
3) Quando a mudanca estiver pronta para merge na `main`, rode os testes de lint e pytest localmente:
```bash
ruff check . | pytest -q
```
4) Se os testes passarem, faça push e abra o Pull Request no GitHub.
5) O Pull Request precisa ser validado (atualmente por um maintainer do codigo, mas no futuro por mais) e obrigatoriamente passar em todos os checks.

## Observacoes e dicas
- Se uma aba nao tiver tabela valida, ela sera ignorada automaticamente.
- Para o modo temporal, a coluna de data precisa ser reconhecida ou selecionada manualmente.
- O modo por ponto agrega por data e permite comparar NA/NO entre pontos no mesmo eixo X.
>>>>>>> origin/main

## Documentacao das funcoes principais
- `build_dataset_from_excel`: abre o Excel, ignora abas de grafico, extrai tabelas e devolve `df_dict` com avisos/erros. (`app/services/dataset_service.py`)
- `clean_basic`: remove colunas vazias, normaliza nomes e tenta converter datas/numeros. (`app/services/dataset_service.py`)
- `drop_month_year_if_full_date_exists`: remove colunas Mes/Ano redundantes quando ha data completa. (`app/services/dataset_service.py`)
- `prep_vol_bombeado` / `prep_vol_infiltrado`: normaliza datas, converte volume e identifica o poco. (`app/pages/create_graph.py`)
- `prep_na_semanal`: interpreta NA/NO/FL, separando valores numericos e status qualitativos, e cria chaves de entidade. (`app/pages/create_graph.py`)
- `build_point_series`: agrega por data e ponto, monta colunas para NA/NO/FL e volume bombeado. (`app/pages/create_graph.py`)
- `build_time_chart_plotly`: monta o grafico temporal e aplica transformacoes por serie. (`app/charts/builder.py`)
- `parse_ptbr_number` / `normalize_dates`: utilitarios de numero e data. (`app/services/date_num_prep.py`)

## Desenvolvedores
- Guilherme Rameh - https://github.com/GuilhermeRameh
- Rodrigo Rameh - https://github.com/DigoRameh
