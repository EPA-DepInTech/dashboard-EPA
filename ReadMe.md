# Dashboard EPA - Visualizacao de dados de pocos

![status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)
![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
<!-- ![streamlit](https://img.shields.io/badge/streamlit-1.52.2-red) -->
![license](https://img.shields.io/badge/licenca-a%20definir-lightgrey)

## Visao geral
Aplicacao Streamlit para carregamento de planilhas Excel e criacao de graficos customizaveis a partir de dados de monitoramento. O app le multiplas abas, identifica tabelas validas, normaliza colunas e oferece modos de visualizacao temporal ou por poco, incluindo suporte especial a resultados laboratoriais com status (SECO, FASE LIVRE, < etc).

## ✨ Funcionalidades principais
- Upload de Excel (.xlsx) com validacao e limpeza automatica das abas.
- Visualizacao de tabelas filtradas (remove linhas "Acumulado" e formata datas).
- Criacao de graficos com 1 ou 2 eixos Y, com selecao de parametros.
- Modo temporal (series por poco) e modo por poco (agregacao por categoria).
- Modo laboratorial com tratamento de status e timeline (heatmap).

## 🧭 Arquitetura e fluxo de dados
1) O usuario faz upload do Excel na pagina inicial (`app/app.py`).
2) O servico de dataset processa o arquivo e retorna um dicionario de DataFrames.
3) O app guarda o dataset em `st.session_state` e mostra a tabela.
4) A pagina de graficos (`app/pages/create_graph.py`) permite mapear colunas, filtrar por periodo/pocos e gerar os graficos com Plotly.

## 🗂️ Estrutura do projeto
- `app/app.py`: pagina inicial, upload e preview da tabela.
- `app/pages/create_graph.py`: tela de criacao de graficos e filtros.
- `app/services/dataset_service.py`: leitura e limpeza de planilhas.
- `app/data/transformer.py`: transformacao de planilhas em formato laboratorial.
- `app/charts/builder.py`: construcao dos graficos Plotly.
- `app/core/state.py`: controle de estado no Streamlit.

## 🧪 Detalhes do processamento de Excel
O servico de dataset:
- Ignora abas claramente de grafico (nome ou conteudo) quando existe aba tabular correspondente.
- Tenta encontrar cabecalho e conteudo tabular mesmo em planilhas com layout irregular.
- Remove colunas "Unnamed:*" e linhas/colunas vazias.
- Normaliza nomes de colunas e converte datas automaticamente quando possivel.
- Converte numeros com virgula e remove colunas de mes/ano redundantes quando ha data completa.
- Para arquivos contendo "Historico" no nome, monta um dataset "master" a partir de abas predefinidas e transforma o formato para linhas por amostra.

## 🧫 Modo laboratorial (status)
Quando o dataset contem colunas no padrao `<param>__num` e `<param>__status`, a pagina de graficos habilita um modo especial:
- "Valores dissolvidos": plota apenas resultados numericos, com marcadores para SECO, FASE LIVRE e MISSING.
- "Timeline (status)": heatmap por poco x data com prioridade de status.

## ✅ Requisitos
- Python 3.12
- Dependencias em `requirements.txt`

## 🚀 Como rodar localmente
Crie e ative o ambiente virtual:
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

Instale as dependencias:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Inicie o app:
```bash
streamlit run .\app\app.py
```

O app fica disponivel em `http://localhost:8501`.

## 🧭 Como usar
1) Abra o app e envie um Excel (.xlsx) pela barra lateral.
2) Valide se as abas foram carregadas (avisos e abas ignoradas aparecem na tela).
3) Visualize os dados na tabela principal.
4) Clique em "Criar grafico" para acessar a pagina de graficos.
5) Selecione colunas de data e poco/ponto, parametros e o tipo de grafico.

## 💡 Observacoes e dicas
- Se uma aba nao tiver tabela valida, ela sera ignorada automaticamente.
- Para o modo temporal, a coluna de data precisa ser reconhecida ou selecionada manualmente.
- O modo por poco agrega valores (media, mediana, minimo, maximo, soma).

## 🤝 Desenvolvedores
- Guilherme Rameh - https://github.com/GuilhermeRameh
- Rodrigo Rameh - https://github.com/DigoRameh
