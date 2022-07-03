# Regressão Linear para o ramo imobiliário
 
## Introdução
Machine learning, mais especificamente o campo da modelagem preditiva, preocupa-se principalmente em minimizar o erro de um modelo ou fazer as previsões mais precisas possíveis, em detrimento da explicabilidade. No aprendizado de máquina aplicado, emprestaremos, reutilizaremos e roubaremos algoritmos de muitos campos diferentes, incluindo estatísticas, e os usaremos para esses fins.

Como tal, a regressão linear foi desenvolvida no campo da estatística e é estudada como um modelo para entender a relação entre variáveis ​​numéricas de entrada e saída, mas foi emprestada pelo aprendizado de máquina. É um algoritmo estatístico e um algoritmo de aprendizado de máquina e sua representação é uma equação linear que combina um conjunto específico de valores de entrada (x) cuja solução é a saída prevista para esse conjunto de valores de entrada (y). Como tal, os valores de entrada (x) e o valor de saída são numéricos.

Tal equação atribui um fator de escala para cada valor de entrada ou coluna, chamado de coeficiente e representado pela letra grega maiúscula Beta (B). Um coeficiente adicional também é adicionado, dando à linha um grau de liberdade adicional (por exemplo, movendo-se para cima e para baixo em um gráfico bidimensional) e é freqüentemente chamado de interceptação ou coeficiente de polarização.

Em um problema de regressão simples (um único x e um único y), a forma do modelo seria:

                        y = B0 + B1*x 


Em dimensões superiores, quando temos mais de uma entrada, a linha é chamada de plano ou hiperplano. A representação, portanto, é a forma da equação e os valores específicos usados ​​para os coeficientes (por exemplo, B0 e B1 no exemplo acima).

## Dataset
O conjunto de dados descreve a venda de propriedades residenciais individuais da cidade de Boston, de 2006 a 2010, ele contém 2.930 observações e um grande número de features (23 nominais, 23 ordinais, 14 discretas e 20 contínuas) envolvidas na avaliação do valor dos imóveis, ou seja, são 80 variáveis explicativas.

Geralmente, as 20 features ​​contínuas estão relacionadas com várias dimensões de área para cada imóvel. Além do típico tamanho do lote e da metragem quadrada total da área habitável, outras variáveis ​​mais específicas são quantificadas no conjunto de dados. Medidas da área do porão, área da sala de estar e até mesmo das varandas estão presentes e divididas em categorias individuais com base na qualidade e no tipo.

## Projeto
Agora que o básico da teoria e as informações sobre o dataset foram passadas, vamos colocar a mão na massa.

Nosso projeto será dividido em 3 partes:

* Limpeza do dataset.
* Achar as features mais importantes, e realizar uma regressão OLS, com um R² superior a 85%.
* Stacking

```
# Importar bibliotecas 

# Bibliotecas para manipulação do dataset
import pandas as pd
import numpy as np

# Bibliotecas para plotagem gráfica  
import seaborn as sns
import matplotlib.pyplot as plt

# Biblioteca para análise estatistica
import scipy
from sklearn.pipeline import make_pipeline

# Eliminar os avisoss 
import warnings
warnings.filterwarnings(action = 'ignore')
```

```# Algumas funções úteis 

def plot_bar(df):
  f, ax = plt.subplots(6, 6, figsize=(25, 27))
  num_vars = df.columns[df.dtypes != 'category']
  for i, c in enumerate(num_vars):
      g = sns.boxplot(df[c], ax=ax[i//6, i%6], color='r')
  f.text(0.4, .90, 'Distribution of numerical vars', size=20, weight='bold', color='black')
  plt.show()

def plot_hist(df):
  f, ax = plt.subplots(6, 6, figsize=(25, 30))
  num_features = df.columns[df.dtypes != 'category']
  for i, c in enumerate(num_features):
      g = sns.histplot(df[c], ax=ax[i//6, i%6], color='r')
  f.text(0.4, .90, 'Distribution of numerical vars', size=20, weight='bold', color='black')
  plt.show()```

