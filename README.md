# Stroke Prediction

Um mini-projeto de Machine Learning desenvolvido com Python, Pandas e Scikit-learn para prever a probabilidade de um paciente sofrer um derrame (stroke) com base em variáveis clínicas e demográficas.

## Motivação

Este projeto foi construído com o objetivo principal de **aprender e aplicar conceitos práticos de Inteligência Artificial e Machine Learning**.

## Tecnologias Utilizadas

- **Python 3.x**
- **Pandas**: Manipulação e análise exploratória dos dados.
- **Scikit-learn**: Construção dos pipelines de pré-processamento, criação e avaliação dos modelos preditivos.
- **Jupyter Notebook**: Para a etapa de Análise Exploratória de Dados (EDA).

## Estrutura do Projeto

O projeto está organizado da seguinte forma para separar responsabilidades e evitar o anti-pattern de "Jupyter Notebooks gigantes e bagunçados":

```
stroke-prediction/
│
├── data/
│   └── healthcare-dataset-stroke-data.csv   # Dataset original
│
├── notebooks/
│   └── EDA.ipynb                            # Análise Exploratória dos Dados
│
├── src/
│   ├── main.py                              # Orquestrador do fluxo do modelo
│   ├── preprocessing.py                     # Pipelines de tratamento (Missing values, Scaling, Encoding)
│   ├── models.py                            # Definição e configuração dos modelos (ex: KNN)
│   └── evaluate.py                          # Métricas e avaliação do modelo
│
└── README.md
```

## Como Executar

1. Certifique-se de ter as bibliotecas necessárias instaladas.
2. Navegue até a pasta `src/`:
   ```bash
   cd src/
   ```
3. Execute o script principal:
   ```bash
   python main.py
   ```

## Roadmap (Em desenvolvimento)

- [ ] Lidar com o desbalanceamento de classes, algo comum em dados médicos.
- [ ] Testar modelos baseados em árvores (Random Forest, XGBoost, etc.)
- [ ] Refinar o tratamento de valores faltantes (por exemplo, explorar mediana ao invés de média para o IMC)
- [ ] Implementar cross-validation
