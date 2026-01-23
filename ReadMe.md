# Dashboard EPA - Visualizacao de dados de pocos

![status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)
<!-- ![streamlit](https://img.shields.io/badge/streamlit-1.52.2-red) -->
![license](https://img.shields.io/badge/licenca-all%20rights%20reserved-blue)

## Visao geral
Aplicacao Streamlit para carregamento de planilhas Excel e criacao de graficos customizaveis a partir de dados de monitoramento. O app le multiplas abas, identifica tabelas validas, normaliza colunas e oferece modos de visualizacao temporal ou por pouco/ponto, com tratamento especial para NA/NO/FL e status qualitativos.

## Funcionalidades principais
- Upload de Excel (.xlsx) com validacao e limpeza automatica das abas.
- Visualizacao de tabelas filtradas (remove linhas "Acumulado" e formata datas).
- Criacao de graficos com 1 ou 2 eixos Y, com selecao de parametros.
- Visualizacao por ponto/poco com NA/NO em barras e volume bombeado em linha.
- Tratamento de valores qualitativos: "Nao medido", "nd", "Destruido", "Soterrado", etc.
- Fase livre qualitativa (Odor, Oleoso, Pelicula) com marcadores dedicados.

## Arquitetura e fluxo de dados
1) O usuario faz upload do Excel na pagina inicial (`app/app.py`).
2) O servico de dataset processa o arquivo e retorna um dicionario de DataFrames.
3) O app guarda o dataset em `st.session_state` e mostra o status de carregamento.
4) A pagina de graficos (`app/pages/create_graph.py`) normaliza datasets, agrega por data e ponto e gera os graficos com Plotly.

## Estrutura do projeto
- `app/app.py`: pagina inicial, upload e preview da tabela.
- `app/pages/create_graph.py`: tela de criacao de graficos e filtros.
- `app/services/dataset_service.py`: leitura e limpeza de planilhas.
- `app/charts/builder.py`: construcao dos graficos Plotly.
- `app/core/state.py`: controle de estado no Streamlit.

## Detalhes do processamento de Excel
O servico de dataset:
- Ignora abas claramente de grafico (nome ou conteudo) quando existe aba tabular correspondente.
- Tenta encontrar cabecalho e conteudo tabular mesmo em planilhas com layout irregular.
- Remove colunas "Unnamed:*" e linhas/colunas vazias.
- Normaliza nomes de colunas e converte datas automaticamente quando possivel.
- Converte numeros com virgula e remove colunas de mes/ano redundantes quando ha data completa.
- Interpreta NA/NO/FL com tratamento de numericos e qualitativos (inclusive "nm" como "Nao medido").

## Requisitos
- Python 3.12
- Dependencias em `requirements.txt`

## Como rodar localmente
Crie e ative o ambiente virtual:
```bash
python -m venv .venv
.\.venv\Scriptsctivate
```

Instale as dependencias:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Inicie o app:
```bash
streamlit run .pppp.py
```

O app fica disponivel em `http://localhost:8501`.

## Como usar
1) Abra o app e envie um Excel (.xlsx) pela barra lateral.
2) Valide se as abas foram carregadas (avisos e abas ignoradas aparecem na tela).
3) Visualize os dados na tabela principal.
4) Clique em "Criar grafico" para acessar a pagina de graficos.
5) Selecione colunas de data e poco/ponto, parametros e o tipo de grafico.

## Observacoes e dicas
- Se uma aba nao tiver tabela valida, ela sera ignorada automaticamente.
- Para o modo temporal, a coluna de data precisa ser reconhecida ou selecionada manualmente.
- O modo por ponto agrega por data e permite comparar NA/NO entre pontos no mesmo eixo X.

## Documentacao das funcoes principais
- `build_dataset_from_excel`: abre o Excel, ignora abas de grafico, extrai tabelas e devolve `df_dict` com avisos/erros. (`app/services/dataset_service.py`)
- `clean_basic`: remove colunas vazias, normaliza nomes e tenta converter datas/numeros. (`app/services/dataset_service.py`)
- `prep_vol_bombeado` / `prep_vol_infiltrado`: normaliza datas, converte volume e identifica o ponto/poco. (`app/pages/create_graph.py`)
- `prep_na_semanal`: interpreta NA/NO/FL, separando valores numericos e status qualitativos, e cria chaves de ponto. (`app/pages/create_graph.py`)
- `build_point_series`: agrega por data e ponto, monta as colunas para NA/NO, fase livre e volume bombeado. (`app/pages/create_graph.py`)
- `build_time_chart_plotly`: monta o grafico temporal e aplica transformacoes por serie. (`app/charts/builder.py`)

## Desenvolvedores
- Guilherme Rameh - https://github.com/GuilhermeRameh
- Rodrigo Rameh - https://github.com/DigoRameh
