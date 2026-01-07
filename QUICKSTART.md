# 🚀 Guia Rápido de Uso

## Instalação Rápida

```bash
# 1. Clone o repositório
git clone https://github.com/fbreseghello/Liver-Disease-Analysis-CNN.git
cd Liver-Disease-Analysis-CNN

# 2. Crie um ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # Linux/Mac

# 3. Instale as dependências
pip install -r requirements.txt
```

## Uso do Notebook

```bash
# Inicie o Jupyter
jupyter notebook

# Abra o arquivo:
# liver-disease-analysis.ipynb
```

## Uso do Script de Treinamento

```bash
# Treinamento básico
python train_model.py

# Treinamento personalizado
python train_model.py --trials 200 --test-size 0.25

# Ver todas as opções
python train_model.py --help
```

## Uso dos Módulos Python

```python
# Exemplo de uso dos módulos
from src import preprocessing, models, visualization, utils
from src.config import *

# 1. Carregar dados
df = utils.load_data('data/HepatitisCdata.csv', index_col=0)

# 2. Pré-processar
X, y = utils.split_features_target(df, 'Category')
preprocessor = preprocessing.DataPreprocessor()
X_transformed = preprocessor.fit_transform(
    X, 
    NUMERIC_FEATURES, 
    CATEGORICAL_FEATURES
)

# 3. Dividir dados
X_train, X_test, y_train, y_test = utils.create_train_test_split(
    X_transformed, y, test_size=0.2
)

# 4. Balancear com SMOTE
X_bal, y_bal = utils.apply_smote(X_train, y_train)

# 5. Treinar modelo (após otimização com Optuna)
builder = models.ModelBuilder()
model = builder.create_model_from_params(best_params)
model.fit(X_bal, y_bal)

# 6. Avaliar
results = builder.evaluate_model(model, X_test, y_test)
models.print_evaluation_results(results)

# 7. Visualizar
visualization.plot_confusion_matrix(y_test, results['predictions'])
```

## Estrutura de Arquivos Esperada

```
Liver-Disease-Analysis-CNN/
├── data/
│   └── HepatitisCdata.csv  (necessário)
├── notebooks/
│   └── liver-disease-analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── visualization.py
│   └── utils.py
├── models/  (criado após treinamento)
├── outputs/  (criado após treinamento)
├── requirements.txt
└── train_model.py
```

## Comandos Úteis

### Verificar instalação
```bash
python -c "import numpy, pandas, sklearn, optuna, shap; print('✓ Tudo instalado!')"
```

### Atualizar dependências
```bash
pip install --upgrade -r requirements.txt
```

### Limpar cache
```bash
# Windows
del /s /q __pycache__
rmdir /s /q .ipynb_checkpoints

# Linux/Mac
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
```

## Solução de Problemas Comuns

### Erro de importação
```bash
# Certifique-se de estar no diretório correto
cd Liver-Disease-Analysis-CNN

# Reinstale as dependências
pip install -r requirements.txt --force-reinstall
```

### Erro de memória no SMOTE
```python
# Reduza k_neighbors
X_bal, y_bal = utils.apply_smote(X_train, y_train, k_neighbors=3)
```

### Optuna muito lento
```python
# Reduza o número de trials
python train_model.py --trials 50
```

## Próximos Passos

1. ✅ Execute o notebook completo
2. ✅ Experimente o script de treinamento
3. ✅ Ajuste hiperparâmetros no config.py
4. ✅ Crie suas próprias visualizações
5. ✅ Adicione novos modelos

## Suporte

Para problemas ou dúvidas:
- Abra uma [Issue no GitHub](https://github.com/fbreseghello/Liver-Disease-Analysis-CNN/issues)
- Consulte a documentação nos módulos Python

---
