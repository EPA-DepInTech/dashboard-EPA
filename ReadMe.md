# Dashboard EPA - Visualização de dados de poços

![status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)
![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
<!-- ![streamlit](https://img.shields.io/badge/streamlit-1.52.2-red) -->
![license](https://img.shields.io/badge/licenca-a%20definir-lightgrey)

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
- `app/core/state.py`: controle de estado no Streamlit.

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
- Python 3.12
- Dependências em `requirements.txt`

## 🚀 Como rodar localmente
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
streamlit run .\app\app.py
```

O app fica disponível em `http://localhost:8501`.

## 🧭 Como usar
1) Abra o app e envie um Excel (.xlsx) pela barra lateral.
2) Valide se as abas foram carregadas (avisos e abas ignoradas aparecem na tela).
3) Visualize os dados na tabela principal.
4) Clique em "Criar grafico" para acessar a página de gráficos.
5) Selecione colunas de data e poço/ponto, parâmetros e o tipo de gráfico.

## 💡 Observações e dicas
- Se uma aba não tiver tabela válida, ela será ignorada automaticamente.
- Para o modo temporal, a coluna de data precisa ser reconhecida ou selecionada manualmente.
- O modo por poço agrega valores (média, mediana, mínimo, máximo, soma).

## 🤝 Desenvolvedores
- Guilherme Rameh - https://github.com/GuilhermeRameh
- Rodrigo Rameh - https://github.com/DigoRameh
