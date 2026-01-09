# Dashboard de visualização customizável de dados de poços

## Objetivo

O objetivo deste projeto é criar um dashboard que permita a criação de visualizações extremamente customizáveis em gráficos e informações referentes a medições de poços de monitoramento e de poços de remediação.

## Hierarquia dos dados
```
| Área
├── Poços
│   ├── Parâmetros
```

## Visualizações padrão

Todos os dados "acumulados" serão divididos por área, não tendo interesse de analisar o conjunto acumulado de todas as áreas.

Dentro de cada área, é possível observar o conjunto somado de todos os poços ou explorar cada poço individualmente (ou qualquer combinação de poços desejados).

Gráficos poderão ser gerados escolhendo os parâmetros que se deseja colocar em conjunto em uma visualização única, limitado pela escala em comum (Ex.: Volume bombeado acumulado junto com NA - Nível da água em escala temporal, ao longo de dois meses).

## Primeiros Passos

Para trabalhar e desenvolver este projeto, primeiro é necessário que o desenvolvedor tenha Python 3.12 instalado.

Para desenvolver dentro de um ambiente virtual, crie um usando o comando:
``` bash
$ python -m venv .venv
```

Ative o ambiente virtual:
``` bash
$ .\.venv\Scripts\activate
```

Instale as dependências:
``` bash
$ python -m pip install --upgrade pip
$ pip install -r requirements.txt
```

## Testando o aplicativo

Após realizar as mudanças desejadas, para testar o aplicativo localmente execute o comando:

``` bash
$ streamlit run .\app\app.py
```

Isso abrirá o serviço localmente, podendo ser acessado pelo *localhost:8501* no navegador de sua escolha.


## Desenvolvedores

- Guilherme Rameh - https://github.com/GuilhermeRameh
- Rodrigo Rameh - https://github.com/DigoRameh