# Liver Disease Analysis - Project

Este projeto foi modernizado e reestruturado em Janeiro de 2026.

## Principais Melhorias

### 🔄 Atualizações Realizadas

1. **README Completo**
   - Documentação detalhada em português
   - Badges informativos
   - Estrutura clara do projeto
   - Instruções de instalação e uso

2. **Código Modernizado**
   - APIs atualizadas do scikit-learn
   - Uso de `suggest_float` ao invés de `suggest_uniform` (Optuna)
   - Melhor tratamento de hiperparâmetros
   - Configurações otimizadas

3. **Estrutura Modular**
   ```
   src/
   ├── __init__.py          # Pacote Python
   ├── config.py            # Configurações centralizadas
   ├── preprocessing.py     # Pré-processamento de dados
   ├── models.py            # Construção e avaliação de modelos
   ├── visualization.py     # Funções de visualização
   └── utils.py             # Utilidades gerais
   ```

4. **Gerenciamento de Dependências**
   - `requirements.txt` com versões atualizadas
   - Compatibilidade com Python 3.9+
   - Bibliotecas modernas de ML

5. **Melhores Práticas**
   - `.gitignore` completo para projetos Python/ML
   - Organização de diretórios
   - Código documentado com docstrings
   - Type hints para melhor legibilidade

### 📁 Nova Estrutura de Diretórios

- `data/` - Datasets originais
- `notebooks/` - Notebooks Jupyter (mover os notebooks aqui)
- `src/` - Código fonte modular
- `models/` - Modelos salvos
- `outputs/` - Resultados e gráficos

### 🚀 Próximos Passos Sugeridos

1. Mover os notebooks para a pasta `notebooks/`
2. Atualizar os notebooks para usar os módulos do `src/`
3. Criar testes unitários em `tests/`
4. Adicionar CI/CD com GitHub Actions
5. Criar um script de treinamento standalone
6. Adicionar logs estruturados

### 📝 Como Usar os Novos Módulos

```python
# Importar módulos
from src import preprocessing, models, visualization, utils
from src.config import *

# Carregar dados
df = utils.load_data(HEPATITIS_DATA, index_col=0)

# Preprocessar
preprocessor = preprocessing.DataPreprocessor(random_state=RANDOM_STATE)
X_transformed = preprocessor.fit_transform(X, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

# Construir modelo
builder = models.ModelBuilder(random_state=RANDOM_STATE)
model = builder.create_model_from_params(best_params)

# Avaliar
results = builder.evaluate_model(model, X_test, y_test, target_names=list(TARGET_LABELS.values()))
models.print_evaluation_results(results)

# Visualizar
visualization.plot_confusion_matrix(y_test, y_pred, labels=list(TARGET_LABELS.values()))
```

### 🔧 Melhorias Técnicas no Notebook

- Imports organizados e agrupados
- Mensagens de confirmação de carregamento
- Uso de constantes para random_state
- Progress bar no Optuna
- Melhor logging
- Código mais limpo e profissional

---

**Desenvolvido com ❤️ | Janeiro 2026**
