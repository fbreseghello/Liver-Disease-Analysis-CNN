# 🏥 Liver Disease Analysis - Machine Learning Project

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Sobre o Projeto

Este projeto utiliza técnicas avançadas de Machine Learning para análise e predição de doenças hepáticas (Hepatite C, Fibrose e Cirrose) a partir de dados clínicos de pacientes. O objetivo é criar um modelo preditivo capaz de realizar detecção precoce dessas condições.

### 🎯 Características Principais

- **Análise Exploratória de Dados (EDA)** completa e detalhada
- **Balanceamento de dados** usando técnica SMOTE
- **Otimização de hiperparâmetros** com Optuna
- **Ensemble Learning** com múltiplos classificadores
- **Interpretabilidade** do modelo usando SHAP
- **Visualizações** interativas e informativas

## 🗂️ Estrutura do Projeto

```
Liver-Disease-Analysis-CNN/
├── data/                           # Datasets
│   ├── HepatitisCdata.csv         # Dataset principal
│   ├── heart.csv                  # Dataset secundário
│   └── healthcare-dataset-stroke-data.csv
├── notebooks/                      # Notebooks Jupyter
│   ├── liver-disease-analysis.ipynb
│   ├── HeartDieseasePrediction.ipynb
│   └── StrokePrediction.ipynb
├── src/                           # Código fonte modular
│   ├── preprocessing.py           # Pré-processamento de dados
│   ├── models.py                  # Definições de modelos
│   ├── visualization.py           # Funções de visualização
│   └── utils.py                   # Utilitários gerais
├── requirements.txt               # Dependências do projeto
├── .gitignore                     # Arquivos ignorados pelo Git
└── README.md                      # Este arquivo

```

## 🚀 Como Usar

### Pré-requisitos

- Python 3.9 ou superior
- pip (gerenciador de pacotes Python)

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/fbreseghello/Liver-Disease-Analysis-CNN.git
cd Liver-Disease-Analysis-CNN
```

2. Crie um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

### Executando o Projeto

Abra o notebook principal:
```bash
jupyter notebook notebooks/liver-disease-analysis.ipynb
```

## 📊 Dataset

O dataset utilizado contém informações clínicas de 615 pacientes, incluindo:

- **Categoria**: Doador de sangue, Hepatite, Fibrose, Cirrose
- **Idade**: idade do paciente em anos
- **Sexo**: sexo do paciente
- **Marcadores Bioquímicos**: ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT

**Fonte**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/HCV+data)

## 🧪 Metodologia

1. **Análise Exploratória (EDA)**
   - Análise univariada, bivariada e multivariada
   - Visualização de distribuições e correlações
   - Análise de componentes principais (PCA)

2. **Pré-processamento**
   - Tratamento de valores faltantes com KNN Imputer
   - Normalização com StandardScaler
   - One-Hot Encoding para variáveis categóricas

3. **Balanceamento**
   - Aplicação da técnica SMOTE para balancear classes

4. **Modelagem**
   - Ensemble com Logistic Regression, KNN, SVM, Random Forest e Naive Bayes
   - Otimização de hiperparâmetros com Optuna (100 trials)

5. **Avaliação**
   - Métricas de desempenho no conjunto de teste
   - Análise de interpretabilidade com SHAP

## 📈 Resultados

Os resultados detalhados da análise e métricas de performance estão disponíveis no notebook principal.

## ⚠️ Aviso Importante

**ATENÇÃO**: Este é um projeto educacional não validado para uso clínico. Não tome decisões médicas baseadas nesses resultados. Sempre consulte um médico especialista.

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fazer um Fork do projeto
2. Criar uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abrir um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👤 Autor

**Felipe Breseghello**
- GitHub: [@fbreseghello](https://github.com/fbreseghello)

## 📚 Referências

- Centers for Disease Control and Prevention (CDC) - [Viral Hepatitis](https://www.cdc.gov/hepatitis/hcv/index.htm)
- UCI Machine Learning Repository - [HCV Dataset](https://archive.ics.uci.edu/ml/datasets/HCV+data)

---

Desenvolvido com ❤️ para fins educacionais e de pesquisa
