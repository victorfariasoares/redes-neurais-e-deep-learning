# Regressão - Previsão do valor de um ativo no dia seguinte

## Autores
- Felipe Bakowski Nantes de Souza  
- Vinicius Grecco Fonseca Mulato  
- Victor Soares

# 1.    Data set - Seleção

Usamos a biblioteca **yfinance** para baixar dados históricos de ações da Apple (AAPL) dos últimos 10 anos. O dataset contém dados diários de preços (abertura, máxima, mínima, fechamento) e volume negociado.


```python
from scipy.stats import normaltest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
from mlp import mlp 

# baixar dados da Apple dos últimos 10 anos
df = yf.download("AAPL", period="10y", interval="1d", auto_adjust=True)
print("Shape:", df.shape)
df.head()
```

    [*********************100%***********************]  1 of 1 completed

    Shape: (2515, 5)


    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Price</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Ticker</th>
      <th>AAPL</th>
      <th>AAPL</th>
      <th>AAPL</th>
      <th>AAPL</th>
      <th>AAPL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-10-26</th>
      <td>25.906366</td>
      <td>26.546834</td>
      <td>25.825465</td>
      <td>26.535599</td>
      <td>265335200</td>
    </tr>
    <tr>
      <th>2015-10-27</th>
      <td>25.742313</td>
      <td>26.189517</td>
      <td>25.616466</td>
      <td>25.933330</td>
      <td>279537600</td>
    </tr>
    <tr>
      <th>2015-10-28</th>
      <td>26.803019</td>
      <td>26.809762</td>
      <td>26.081650</td>
      <td>26.277162</td>
      <td>342205600</td>
    </tr>
    <tr>
      <th>2015-10-29</th>
      <td>27.086176</td>
      <td>27.122133</td>
      <td>26.578296</td>
      <td>26.674928</td>
      <td>204909200</td>
    </tr>
    <tr>
      <th>2015-10-30</th>
      <td>26.854704</td>
      <td>27.241232</td>
      <td>26.843467</td>
      <td>27.189545</td>
      <td>197461200</td>
    </tr>
  </tbody>
</table>
</div>




```python
# remover nível de ticker se presente (robusto)
if isinstance(df.columns, pd.MultiIndex):
    for lvl in range(df.columns.nlevels):
        # se a camada for constante (ex.: todos 'AAPL'), removemos essa camada
        if df.columns.get_level_values(lvl).nunique() == 1:
            df.columns = df.columns.droplevel(lvl)
            break
print("Columns after flattening:", df.columns.tolist())
```

    Columns after flattening: ['Close', 'High', 'Low', 'Open', 'Volume']


# 2. Data set - Explicação

O `DataFrame` `df` reúne cotações diárias da Apple (ticker `AAPL`) obtidas via `yfinance` para os últimos 10 anos (`period="10y"`, `interval="1d"`). A coluna de índice (`Date`) marca cada pregão. A partir das séries básicas (`Open`, `High`, `Low`, `Close`, `Volume`) derivaram‑se as features abaixo:

- `Price_Range`: `High - Low`. Medida de volatilidade intraday.  
- `Price_Change`: `Close - Open`. Variação diária absoluta (momentum do dia).  
- `High_Low_Ratio`: `High / Low`. Proporção de amplitude do dia (volatilidade relativa).  
- `Return_1d`, `Return_3d`, `Return_5d`, `Return_10d`: retornos percentuais em janelas 1/3/5/10 dias. Capturam momentum em diferentes horizontes.  
- `MA_5`, `MA_10`, `MA_20`, `MA_50`: médias móveis do preço de fechamento (suavização de tendência).  
- `Volatility_5`, `Volatility_10`, `Volatility_20`: desvio padrão móvel do fechamento (risco/instabilidade em janelas distintas).  
- `Volume_MA_5`, `Volume_MA_10`: médias móveis do volume (interesse de mercado suavizado).  
- `BB_Middle`, `BB_Upper`, `BB_Lower`, `BB_Width`: bandas de Bollinger (MA_20 ± 2·Volatility_20) e largura (volatilidade implícita).  
- `RSI`: índice de força relativa (14 dias), indicador de sobrecompra/sobrevenda.  
- `MACD`, `MACD_Signal`, `MACD_Histogram`: MACD (EMA12−EMA26), sua linha de sinal (EMA9) e histograma (força do momentum).  
- `Momentum_5`, `Momentum_10`: diferença absoluta do fechamento em 5 e 10 dias (velocidade da variação).  
- `ROC_5`, `ROC_10`: rate of change (%) em 5 e 10 dias (retorno percentual em diferentes janelas).  
- `Target`: `Close.shift(-1)` — preço de fechamento do próximo dia (variável alvo).

Essas features procuram capturar preço, volume, volatilidade e momentum em múltiplas escalas temporais. O objetivo é treinar uma rede neural capaz de estimar o preço de fechamento (`Close`) do dia seguinte a partir desse histórico.

Ainda, vale notar que todos os dados (features + target) são numéricos continuos.

## Criação de novas features


```python
print("\n CRIANDO FEATURES AVANÇADAS...")

# Features baseadas em preço
df['Price_Range'] = df['High'] - df['Low']  # Volatilidade intraday
df['Price_Change'] = df['Close'] - df['Open']  # Mudança diária
df['High_Low_Ratio'] = df['High'] / df['Low']  # Razão high/low

# Retornos percentuais
df['Return_1d'] = df['Close'].pct_change(1)  # Retorno de 1 dia
df['Return_3d'] = df['Close'].pct_change(3)  # Retorno de 3 dias
df['Return_5d'] = df['Close'].pct_change(5)  # Retorno de 5 dias
df['Return_10d'] = df['Close'].pct_change(10)  # Retorno de 10 dias

# Médias móveis
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_10'] = df['Close'].rolling(window=10).mean()
df['MA_20'] = df['Close'].rolling(window=20).mean()
df['MA_50'] = df['Close'].rolling(window=50).mean()

# Desvio padrão móvel (volatilidade)
df['Volatility_5'] = df['Close'].rolling(window=5).std()
df['Volatility_10'] = df['Close'].rolling(window=10).std()
df['Volatility_20'] = df['Close'].rolling(window=20).std()

# Média móvel do volume
df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()

# Bandas de Bollinger (simplificadas)
df['BB_Middle'] = df['MA_20']
df['BB_Upper'] = df['MA_20'] + 2 * df['Volatility_20']
df['BB_Lower'] = df['MA_20'] - 2 * df['Volatility_20']
df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

# RSI simplificado (Relative Strength Index)
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD (Moving Average Convergence Divergence)
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

# Features de momentum
df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
df['Momentum_10'] = df['Close'] - df['Close'].shift(10)

# Rate of Change (ROC)
df['ROC_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100

# Target: Preço de fechamento do próximo dia
df['Target'] = df['Close'].shift(-1)

print(f"Shape final após feature engineering: {df.shape}")
print(f"Total de features criadas: {df.shape[1] - 3}")  # -3 para Date, Close original e Target
```

    
     CRIANDO FEATURES AVANÇADAS...
    Shape final após feature engineering: (2515, 34)
    Total de features criadas: 31


## Crescimento da ação ao longo do tempo


```python
plt.figure(figsize=(14,5))
plt.plot(df.index, df['Close'], label='Close Price')
plt.title('Apple Stock - Close Price (10 anos)')
plt.xlabel('Data')
plt.ylabel('Preço (USD)')
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](regression_files/regression_6_0.png)
    



```python
plt.figure(figsize=(14,5))
plt.plot(df.index, df['Volume'], label='Close Price')
plt.title('Apple Stock - Close Price (10 anos)')
plt.xlabel('Data')
plt.ylabel('Volume (USD)')
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](regression_files/regression_7_0.png)
    


## Colunas com valores faltando


```python
print("Valores faltantes por coluna:\n", df.isnull().sum())
df = df.dropna()
```

    Valores faltantes por coluna:
     Price
    Close              0
    High               0
    Low                0
    Open               0
    Volume             0
    Price_Range        0
    Price_Change       0
    High_Low_Ratio     0
    Return_1d          1
    Return_3d          3
    Return_5d          5
    Return_10d        10
    MA_5               4
    MA_10              9
    MA_20             19
    MA_50             49
    Volatility_5       4
    Volatility_10      9
    Volatility_20     19
    Volume_MA_5        4
    Volume_MA_10       9
    BB_Middle         19
    BB_Upper          19
    BB_Lower          19
    BB_Width          19
    RSI               13
    MACD               0
    MACD_Signal        0
    MACD_Histogram     0
    Momentum_5         5
    Momentum_10       10
    ROC_5              5
    ROC_10            10
    Target             1
    dtype: int64



```python
print("Valores faltantes por coluna:\n", df.isnull().sum())
```

    Valores faltantes por coluna:
     Price
    Close             0
    High              0
    Low               0
    Open              0
    Volume            0
    Price_Range       0
    Price_Change      0
    High_Low_Ratio    0
    Return_1d         0
    Return_3d         0
    Return_5d         0
    Return_10d        0
    MA_5              0
    MA_10             0
    MA_20             0
    MA_50             0
    Volatility_5      0
    Volatility_10     0
    Volatility_20     0
    Volume_MA_5       0
    Volume_MA_10      0
    BB_Middle         0
    BB_Upper          0
    BB_Lower          0
    BB_Width          0
    RSI               0
    MACD              0
    MACD_Signal       0
    MACD_Histogram    0
    Momentum_5        0
    Momentum_10       0
    ROC_5             0
    ROC_10            0
    Target            0
    dtype: int64


Foi aplicado o método **`dropna()`** para remover as linhas com valores ausentes, pois algumas **features derivadas** (como *Momentum5*, *MA_5*, *Volatility_10*, *Return_10d*, etc.) apresentam valores faltantes nas primeiras observações.

Por exemplo, o cálculo do **Momentum5** requer os cinco dias anteriores; portanto, **as quatro primeiras linhas não têm valor definido** e retornam *NaN*. O mesmo ocorre para outros indicadores que utilizam períodos maiores (como *MA_20* ou *Return_10d*).



## Visualizando as features


```python
# Configura grid de subplots
n_cols = 4
n_features = len(df.columns)
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    ax = axes[i]
    # remove NaNs antes de plotar
    data = df[col].dropna()
    if data.size == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title(col)
        ax.set_axis_off()
        continue
    sns.histplot(data, bins=50, kde=True, ax=ax, stat='density', color='tab:blue')
    ax.set_title(col)
    ax.set_xlabel('')
    ax.set_ylabel('Density')

# Desativa eixos extras se houver
for j in range(n_features, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()
```


    
![png](regression_files/regression_13_0.png)
    



```python
pd.options.display.float_format = '{:.3f}'.format

corr_df = df.corr()

order = corr_df['Target'].abs().sort_values(ascending=False).index.tolist()
# opcional: colocar o target no final para melhor leitura
order = [c for c in order if c != 'Target'] + ['Target']
corr_df = corr_df.loc[order, order]

# Plot heatmap com layout melhorado
plt.figure(figsize=(14, 12), dpi=120)
sns.heatmap(
    corr_df,
    cmap='coolwarm',
    vmin=-1, vmax=1,
    annot=True,
    fmt='.2f',
    annot_kws={'size':8},
    linewidths=0.25,
    cbar_kws={'shrink':0.6, 'label':'Pearson r'}
)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.title('Correlation matrix (features and target)', fontsize=14)
plt.tight_layout()
plt.show()

top = corr_df['Target'].drop(index='Target').sort_values(key=lambda s: s.abs(), ascending=False)
print('\nTop features por correlação (com sinal) com Target:')
display(top.to_frame(name='corr_with_target'))

# Estatísticas descritivas: média e desvio padrão
stats = df.agg(['mean', 'std']).T
stats = stats.rename(columns={'mean':'mean','std':'std'})
print('\nMédia e desvio padrão das variáveis numéricas:')
display(stats.style.format('{:.4f}'))
```


    
![png](regression_files/regression_14_0.png)
    


    
    Top features por correlação (com sinal) com Target:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>corr_with_target</th>
    </tr>
    <tr>
      <th>Price</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Close</th>
      <td>0.999</td>
    </tr>
    <tr>
      <th>Low</th>
      <td>0.999</td>
    </tr>
    <tr>
      <th>High</th>
      <td>0.999</td>
    </tr>
    <tr>
      <th>Open</th>
      <td>0.999</td>
    </tr>
    <tr>
      <th>MA_5</th>
      <td>0.999</td>
    </tr>
    <tr>
      <th>MA_10</th>
      <td>0.998</td>
    </tr>
    <tr>
      <th>MA_20</th>
      <td>0.997</td>
    </tr>
    <tr>
      <th>BB_Middle</th>
      <td>0.997</td>
    </tr>
    <tr>
      <th>BB_Upper</th>
      <td>0.996</td>
    </tr>
    <tr>
      <th>BB_Lower</th>
      <td>0.995</td>
    </tr>
    <tr>
      <th>MA_50</th>
      <td>0.992</td>
    </tr>
    <tr>
      <th>Volatility_20</th>
      <td>0.763</td>
    </tr>
    <tr>
      <th>BB_Width</th>
      <td>0.763</td>
    </tr>
    <tr>
      <th>Volatility_10</th>
      <td>0.690</td>
    </tr>
    <tr>
      <th>Price_Range</th>
      <td>0.677</td>
    </tr>
    <tr>
      <th>Volume_MA_10</th>
      <td>-0.670</td>
    </tr>
    <tr>
      <th>Volume_MA_5</th>
      <td>-0.631</td>
    </tr>
    <tr>
      <th>Volatility_5</th>
      <td>0.606</td>
    </tr>
    <tr>
      <th>Volume</th>
      <td>-0.540</td>
    </tr>
    <tr>
      <th>MACD_Signal</th>
      <td>0.200</td>
    </tr>
    <tr>
      <th>MACD</th>
      <td>0.192</td>
    </tr>
    <tr>
      <th>Momentum_10</th>
      <td>0.100</td>
    </tr>
    <tr>
      <th>High_Low_Ratio</th>
      <td>0.097</td>
    </tr>
    <tr>
      <th>Momentum_5</th>
      <td>0.075</td>
    </tr>
    <tr>
      <th>RSI</th>
      <td>-0.053</td>
    </tr>
    <tr>
      <th>Price_Change</th>
      <td>0.051</td>
    </tr>
    <tr>
      <th>MACD_Histogram</th>
      <td>0.018</td>
    </tr>
    <tr>
      <th>ROC_5</th>
      <td>0.011</td>
    </tr>
    <tr>
      <th>Return_5d</th>
      <td>0.011</td>
    </tr>
    <tr>
      <th>Return_3d</th>
      <td>0.008</td>
    </tr>
    <tr>
      <th>ROC_10</th>
      <td>0.008</td>
    </tr>
    <tr>
      <th>Return_10d</th>
      <td>0.008</td>
    </tr>
    <tr>
      <th>Return_1d</th>
      <td>0.003</td>
    </tr>
  </tbody>
</table>
</div>


    
    Média e desvio padrão das variáveis numéricas:



<style type="text/css">
</style>
<table id="T_e67a1">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_e67a1_level0_col0" class="col_heading level0 col0" >mean</th>
      <th id="T_e67a1_level0_col1" class="col_heading level0 col1" >std</th>
    </tr>
    <tr>
      <th class="index_name level0" >Price</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_e67a1_level0_row0" class="row_heading level0 row0" >Close</th>
      <td id="T_e67a1_row0_col0" class="data row0 col0" >111.4397</td>
      <td id="T_e67a1_row0_col1" class="data row0 col1" >70.6981</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row1" class="row_heading level0 row1" >High</th>
      <td id="T_e67a1_row1_col0" class="data row1 col0" >112.5500</td>
      <td id="T_e67a1_row1_col1" class="data row1 col1" >71.3794</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row2" class="row_heading level0 row2" >Low</th>
      <td id="T_e67a1_row2_col0" class="data row2 col0" >110.2107</td>
      <td id="T_e67a1_row2_col1" class="data row2 col1" >69.9294</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row3" class="row_heading level0 row3" >Open</th>
      <td id="T_e67a1_row3_col0" class="data row3 col0" >111.3314</td>
      <td id="T_e67a1_row3_col1" class="data row3 col1" >70.6237</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row4" class="row_heading level0 row4" >Volume</th>
      <td id="T_e67a1_row4_col0" class="data row4 col0" >102733762.8398</td>
      <td id="T_e67a1_row4_col1" class="data row4 col1" >57807883.1952</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row5" class="row_heading level0 row5" >Price_Range</th>
      <td id="T_e67a1_row5_col0" class="data row5 col0" >2.3393</td>
      <td id="T_e67a1_row5_col1" class="data row5 col1" >2.1404</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row6" class="row_heading level0 row6" >Price_Change</th>
      <td id="T_e67a1_row6_col0" class="data row6 col0" >0.1083</td>
      <td id="T_e67a1_row6_col1" class="data row6 col1" >1.9299</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row7" class="row_heading level0 row7" >High_Low_Ratio</th>
      <td id="T_e67a1_row7_col0" class="data row7 col0" >1.0205</td>
      <td id="T_e67a1_row7_col1" class="data row7 col1" >0.0126</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row8" class="row_heading level0 row8" >Return_1d</th>
      <td id="T_e67a1_row8_col0" class="data row8 col0" >0.0012</td>
      <td id="T_e67a1_row8_col1" class="data row8 col1" >0.0184</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row9" class="row_heading level0 row9" >Return_3d</th>
      <td id="T_e67a1_row9_col0" class="data row9 col0" >0.0034</td>
      <td id="T_e67a1_row9_col1" class="data row9 col1" >0.0308</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row10" class="row_heading level0 row10" >Return_5d</th>
      <td id="T_e67a1_row10_col0" class="data row10 col0" >0.0056</td>
      <td id="T_e67a1_row10_col1" class="data row10 col1" >0.0390</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row11" class="row_heading level0 row11" >Return_10d</th>
      <td id="T_e67a1_row11_col0" class="data row11 col0" >0.0111</td>
      <td id="T_e67a1_row11_col1" class="data row11 col1" >0.0547</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row12" class="row_heading level0 row12" >MA_5</th>
      <td id="T_e67a1_row12_col0" class="data row12 col0" >111.2477</td>
      <td id="T_e67a1_row12_col1" class="data row12 col1" >70.5831</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row13" class="row_heading level0 row13" >MA_10</th>
      <td id="T_e67a1_row13_col0" class="data row13 col0" >111.0144</td>
      <td id="T_e67a1_row13_col1" class="data row13 col1" >70.4576</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row14" class="row_heading level0 row14" >MA_20</th>
      <td id="T_e67a1_row14_col0" class="data row14 col0" >110.5510</td>
      <td id="T_e67a1_row14_col1" class="data row14 col1" >70.2154</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row15" class="row_heading level0 row15" >MA_50</th>
      <td id="T_e67a1_row15_col0" class="data row15 col0" >109.2111</td>
      <td id="T_e67a1_row15_col1" class="data row15 col1" >69.5443</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row16" class="row_heading level0 row16" >Volatility_5</th>
      <td id="T_e67a1_row16_col0" class="data row16 col0" >1.6800</td>
      <td id="T_e67a1_row16_col1" class="data row16 col1" >1.6826</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row17" class="row_heading level0 row17" >Volatility_10</th>
      <td id="T_e67a1_row17_col0" class="data row17 col0" >2.3579</td>
      <td id="T_e67a1_row17_col1" class="data row17 col1" >2.1275</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row18" class="row_heading level0 row18" >Volatility_20</th>
      <td id="T_e67a1_row18_col0" class="data row18 col0" >3.3183</td>
      <td id="T_e67a1_row18_col1" class="data row18 col1" >2.6446</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row19" class="row_heading level0 row19" >Volume_MA_5</th>
      <td id="T_e67a1_row19_col0" class="data row19 col0" >102870287.1481</td>
      <td id="T_e67a1_row19_col1" class="data row19 col1" >49493460.4022</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row20" class="row_heading level0 row20" >Volume_MA_10</th>
      <td id="T_e67a1_row20_col0" class="data row20 col0" >102952212.1988</td>
      <td id="T_e67a1_row20_col1" class="data row20 col1" >46586828.2119</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row21" class="row_heading level0 row21" >BB_Middle</th>
      <td id="T_e67a1_row21_col0" class="data row21 col0" >110.5510</td>
      <td id="T_e67a1_row21_col1" class="data row21 col1" >70.2154</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row22" class="row_heading level0 row22" >BB_Upper</th>
      <td id="T_e67a1_row22_col0" class="data row22 col0" >117.1875</td>
      <td id="T_e67a1_row22_col1" class="data row22 col1" >74.3426</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row23" class="row_heading level0 row23" >BB_Lower</th>
      <td id="T_e67a1_row23_col0" class="data row23 col0" >103.9144</td>
      <td id="T_e67a1_row23_col1" class="data row23 col1" >66.2537</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row24" class="row_heading level0 row24" >BB_Width</th>
      <td id="T_e67a1_row24_col0" class="data row24 col0" >13.2731</td>
      <td id="T_e67a1_row24_col1" class="data row24 col1" >10.5783</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row25" class="row_heading level0 row25" >RSI</th>
      <td id="T_e67a1_row25_col0" class="data row25 col0" >56.0671</td>
      <td id="T_e67a1_row25_col1" class="data row25 col1" >17.9194</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row26" class="row_heading level0 row26" >MACD</th>
      <td id="T_e67a1_row26_col0" class="data row26 col0" >0.6328</td>
      <td id="T_e67a1_row26_col1" class="data row26 col1" >2.3923</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row27" class="row_heading level0 row27" >MACD_Signal</th>
      <td id="T_e67a1_row27_col0" class="data row27 col0" >0.6254</td>
      <td id="T_e67a1_row27_col1" class="data row27 col1" >2.2372</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row28" class="row_heading level0 row28" >MACD_Histogram</th>
      <td id="T_e67a1_row28_col0" class="data row28 col0" >0.0075</td>
      <td id="T_e67a1_row28_col1" class="data row28 col1" >0.7491</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row29" class="row_heading level0 row29" >Momentum_5</th>
      <td id="T_e67a1_row29_col0" class="data row29 col0" >0.4770</td>
      <td id="T_e67a1_row29_col1" class="data row29 col1" >5.2611</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row30" class="row_heading level0 row30" >Momentum_10</th>
      <td id="T_e67a1_row30_col0" class="data row30 col0" >0.9298</td>
      <td id="T_e67a1_row30_col1" class="data row30 col1" >7.0770</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row31" class="row_heading level0 row31" >ROC_5</th>
      <td id="T_e67a1_row31_col0" class="data row31 col0" >0.5610</td>
      <td id="T_e67a1_row31_col1" class="data row31 col1" >3.9028</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row32" class="row_heading level0 row32" >ROC_10</th>
      <td id="T_e67a1_row32_col0" class="data row32 col0" >1.1090</td>
      <td id="T_e67a1_row32_col1" class="data row32 col1" >5.4714</td>
    </tr>
    <tr>
      <th id="T_e67a1_level0_row33" class="row_heading level0 row33" >Target</th>
      <td id="T_e67a1_row33_col0" class="data row33 col0" >111.5371</td>
      <td id="T_e67a1_row33_col1" class="data row33 col1" >70.7412</td>
    </tr>
  </tbody>
</table>



# 3. Limpeza & Normalização

### Abordagem geral

Antes de aplicar qualquer técnica de normalização, é essencial **verificar a distribuição dos dados**. A escolha entre **normalização Z-score** ou **Min-Max** depende diretamente da **normalidade** da variável analisada.

1. **Teste de normalidade:**
   Aplicamos o **teste de D’Agostino e Pearson** para avaliar se a variável segue uma distribuição normal.

   * **p-valor alto (p > 0,05):** não há evidências para rejeitar a hipótese de normalidade → aplicar **Z-score**.
   * **p-valor baixo (p ≤ 0,05):** rejeita-se a hipótese de normalidade → aplicar **Min-Max**.





```python
alpha = 0.02

df_orig = df.copy()

# decidir tipo de scaler usando uma amostra do dataset inteiro
sample = df.sample(n=min(1000, len(df)), random_state=42)

applied = {}

for col in df.columns:
    if col != 'Target':
        vals_sample = sample[col].values.reshape(-1, 1).astype(float)

        stat, p = normaltest(vals_sample[:100].ravel())

        if p >= alpha:
            scaler = StandardScaler()
            choice = "zscore"
        else:
            scaler = MinMaxScaler(feature_range=(-1,1))
            choice = "minmax"

        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).ravel()

        applied[col] = choice

cols = list(df.columns)
n_features = len(cols)-1


fig, axes = plt.subplots(n_features, 2, figsize=(12, 2.5 * n_features))
axes = np.atleast_2d(axes)

for i, c in enumerate(cols):
    if c != "Target":
        ax_orig = axes[i, 0]
        ax_scaled = axes[i, 1]

        data_orig = df_orig[c]
        data_scaled = df[c]

        # plot original
        sns.histplot(data_orig, bins=50, kde=True, ax=ax_orig, stat="density", color="tab:blue")
        ax_orig.set_title(f"{c} (original)")
        ax_orig.set_xlabel("")
        ax_orig.set_ylabel("Density")

        # plot scaled
        sns.histplot(data_scaled, bins=50, kde=True, ax=ax_scaled, stat="density", color="tab:orange")
        choice_label = applied.get(c, "?")
        ax_scaled.set_title(f"{c} (scaled: {choice_label})")
        ax_scaled.set_xlabel("")
        ax_scaled.set_ylabel("Density")

plt.tight_layout()
plt.show()

```


    
![png](regression_files/regression_16_0.png)
    


# 4. Implementação MLP



```python
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class mlp:
    def __init__(self, n_features:int, n_hidden_layers: int, n_neurons_per_layer: list, 
                activation: str, loss: str, optimizer: str, epochs: int, eta: float) -> None:
        self.n_features = n_features
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons_per_layer = n_neurons_per_layer
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.eta = eta

        self.weights = []
        self.biases = []
        self.layer_dims = [n_features] + n_neurons_per_layer

        for i in range(len(self.layer_dims) - 1):
            w = np.random.randn(self.layer_dims[i+1], self.layer_dims[i]) * 0.1
            b = np.zeros((self.layer_dims[i+1],))
            self.weights.append(w)
            self.biases.append(b)

        # histórico de treino
        self.metric_name = "Accuracy" if loss == "cross_entropy" else "RMSE"
        self.history = {"loss": [], "metric": []}

        print("\n=== Inicialização de Pesos e Biases ===")
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            print(f"Camada {i+1}:")
            print(f"W{i+1} shape {w.shape}:\n{w}")
            print(f"b{i+1} shape {b.shape}:\n{b}\n")

    def train(self, X, y, threshold: float = 5e-3, window: int = 10) -> None:
        loss_history = []

        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(len(y)):
                y_pred, cache = self.forward_pass(X[i])
                loss = self.loss_calculation(y[i], y_pred)
                total_loss += loss
                grads_w, grads_b = self.backpropagation(y[i], y_pred, cache)
                self.update_parameters(grads_w, grads_b)

            avg_loss = total_loss / len(y)
            loss_history.append(avg_loss)

            preds_train = self.test(X)
            if self.loss == "cross_entropy":
                train_metric = self.calculate_accuracy(y, preds_train)
            else:
                train_metric = self.calculate_rmse(y, preds_train)

            # armazenar no histórico
            self.history["loss"].append(avg_loss)
            self.history["metric"].append(train_metric)

            if epoch % 10 == 0:
                if self.loss == "cross_entropy":
                    metric_str = f"{train_metric*100:.2f}%" if not np.isnan(train_metric) else "nan"
                else:
                    metric_str = f"{train_metric:.6f}"
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, Train {self.metric_name}: {metric_str}")

            # critério de parada: média móvel dos últimos "window" epochs
            if epoch >= window:
                moving_avg_prev = np.mean(loss_history[-2*window:-window])
                moving_avg_curr = np.mean(loss_history[-window:])
                if abs(moving_avg_prev - moving_avg_curr) < threshold:
                    print(f"Treinamento encerrado no epoch {epoch} (convergência detectada).")
                    break

    def test(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for i in range(len(X)):
            y_pred, _ = self.forward_pass(X[i])
            if self.loss == "cross_entropy":
                preds.append(np.argmax(y_pred))
            else:
                val = np.array(y_pred).ravel()
                preds.append(val[0] if val.size > 0 else val)
        return np.array(preds)


    def evaluate(self, X: np.ndarray, y: np.ndarray, plot_confusion: bool, plot_roc: bool, preds: np.ndarray):
        if self.loss == "cross_entropy":
            acc = self.calculate_accuracy(y, preds)
            print(f"Accuracy: {acc*100:.2f}%")
            report = {"accuracy": acc}
            self.last_classification_report = report
        else:
            y_true = np.asarray(y).ravel()
            preds = np.asarray(preds).ravel()
            metrics = self._compute_regression_metrics(y_true, preds)

            baseline_preds = np.full_like(y_true, y_true.mean())
            baseline_metrics = self._compute_regression_metrics(y_true, baseline_preds)

            self._print_regression_summary(metrics, baseline_metrics)

            report = {
                "metrics": metrics,
                "baseline_metrics": baseline_metrics,
                "residuals": y_true - preds,
                "y_true": y_true,
                "y_pred": preds,
            }
            self.last_regression_report = report

        print("\n=== Pesos e Biases do Modelo ===")
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            print(f"\nCamada {i+1}:")
            print(f"  Pesos W{i+1} (shape {w.shape}):")
            print(w)
            print(f"  Biases b{i+1} (shape {b.shape}):")
            print(b)

        binary = (len(np.unique(y)) == 2)

        if plot_confusion and self.loss == "cross_entropy":
            self.plot_confusion_matrix(y, preds)
        elif plot_confusion:
            print("Confusion matrix não é aplicável para regressão.")

        if plot_roc and binary and self.loss == "cross_entropy":
            self.plot_roc_curve(X, y)
        elif plot_roc and self.loss != "cross_entropy":
            print("ROC curve não é aplicável para regressão.")

        # sempre plota histórico se existir
        if self.history and len(self.history.get("loss", [])) > 0:
            self.plot_history()

        return report

    # -------- Funções auxiliares de plot --------
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(6,5))
        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                    xticklabels=np.unique(y_true),
                    yticklabels=np.unique(y_true))
        plt.title("Confusion Matrix (normalized by row)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def plot_roc_curve(self, X, y_true):
        y_scores = []
        for i in range(len(X)):
            y_pred, _ = self.forward_pass(X[i])
            if self.loss == "cross_entropy":
                y_scores.append(y_pred[1])
            else:
                val = np.array(y_pred).ravel()
                y_scores.append(val[0] if val.size > 0 else val)
        y_scores = np.array(y_scores)

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2,
                 label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.show()

    def plot_history(self):
        """Plota loss e accuracy armazenados em self.history."""
        loss = self.history.get("loss", [])
        metric = self.history.get("metric", [])
        epochs = np.arange(1, len(loss)+1)

        fig, axes = plt.subplots(1, 2, figsize=(12,4))
        # loss
        axes[0].plot(epochs, loss, marker="o")
        axes[0].set_title("Loss por Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle="--", alpha=0.4)
        # accuracy
        axes[1].plot(epochs, metric, marker="o")
        axes[1].set_title(f"{self.metric_name} por Epoch (treino)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel(self.metric_name)
        axes[1].grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.show()

    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)

    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        return np.sqrt(np.mean((y_true - y_pred)**2))

    def _compute_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        r2 = r2_score(y_true, y_pred)

        return {
            "MAE": mae,
            "MAPE (%)": mape,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
        }

    def _print_regression_summary(self, model_metrics: dict, baseline_metrics: dict) -> None:
        header = f"{'Metric':<12}{'Model':>14}{'Baseline':>14}"
        print("\nRegression metrics (model vs. mean baseline):")
        print(header)
        print("-" * len(header))
        for key in ["MAE", "MAPE (%)", "MSE", "RMSE", "R2"]:
            model_val = model_metrics.get(key, float("nan"))
            baseline_val = baseline_metrics.get(key, float("nan"))
            print(f"{key:<12}{model_val:>14.6f}{baseline_val:>14.6f}")

    def forward_pass(self, x: np.ndarray) -> tuple:
        a = x
        cache = {"z": [], "a": [a]}
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], a) + self.biases[i]
            if i == len(self.weights) - 1 and self.loss == "cross_entropy":
                a = self.softmax(z)
            else:
                a = self.activation_function(z)
            cache["z"].append(z)
            cache["a"].append(a)
        return a, cache

    def backpropagation(self, y_true: np.ndarray, y_pred: np.ndarray, cache: dict) -> tuple:
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # Última camada
        if self.loss == "cross_entropy":
            delta = self.derive_cross_entropy(y_true, y_pred)
        elif self.loss == "mse":
            dloss_dy_pred = self.derive_mse(y_true, y_pred)
            if self.activation == "sigmoid":
                delta = dloss_dy_pred * self.derive_sigmoid(cache["z"][-1])
            elif self.activation == "tanh":
                delta = dloss_dy_pred * self.derive_tanh(cache["z"][-1])
            elif self.activation == "relu":
                delta = dloss_dy_pred * self.derive_relu(cache["z"][-1])
        else:
            raise ValueError("Loss não suportada")

        grads_w[-1] = np.outer(delta, cache["a"][-2])
        grads_b[-1] = delta

        # Camadas ocultas
        # não faz sentido atualizar delta em "a", já que só faz sentido atualizar os pesos, que estão na pré ativação "z"
        for l in reversed(range(len(self.weights)-1)):
            delta = np.dot(self.weights[l+1].T, delta)
            if self.activation == "sigmoid":
                delta *= self.derive_sigmoid(cache["z"][l])
            elif self.activation == "tanh":
                delta *= self.derive_tanh(cache["z"][l])
            elif self.activation == "relu":
                delta *= self.derive_relu(cache["z"][l])
            grads_w[l] = np.outer(delta, cache["a"][l]) # o primeiro x esta dentro do cache "a"
            grads_b[l] = delta

        return grads_w, grads_b
        
    def update_parameters(self, grads_w, grads_b):
        if self.optimizer == "gd":
            for i in range(len(self.weights)):
                self.weights[i] -= self.eta * grads_w[i]
                self.biases[i]  -= self.eta * grads_b[i]
        else:
            raise ValueError(f"Optimizer {self.optimizer} não suportado")

    def loss_calculation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.loss == 'mse':
            return self.mse(y_true, y_pred)
        elif self.loss == 'cross_entropy':
            return self.cross_entropy(y_true, y_pred)
        else:
            raise ValueError(f"Função de loss {self.loss} não suportada")

    def activation_function(self, z: np.ndarray) -> np.ndarray:
        if self.activation == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation == 'tanh':
            return self.tanh(z)
        elif self.activation == 'relu':
            return self.relu(z)
        else:
            raise ValueError(f"Função de ativação {self.activation} não suportada")
        
    def mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred)**2)

    def derive_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -2*(y_true - y_pred)

    def cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        num_classes = len(y_pred)
        y_true_onehot = np.eye(num_classes)[y_true]
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.sum(y_true_onehot * np.log(y_pred))

    def derive_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        num_classes = len(y_pred)
        y_true_onehot = np.eye(num_classes)[y_true]
        return y_pred - y_true_onehot

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def derive_sigmoid(self, z: np.ndarray) -> np.ndarray:
        s = self.sigmoid(z)
        return s * (1 - s)

    def tanh(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def derive_tanh(self, z: np.ndarray) -> np.ndarray:
        return 1 - (np.tanh(z))**2

    def relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def derive_relu(self, z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    def softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)

```

Essa implementação de MLP permite que o usuário escolha o número de layers e seu tamanho (output também), permite que o usuário escolhe entre Relu, sigmoid ou tanh como função de ativação, disponibiliza cross-entropy ou MSE para calculo de erro, mas tem a limitação de que só permite o usuário escolher o Gradient Descent padrão (GD) como otimização, mas como ele trabalha em modo estocástico (atualiza peso em cada amostra), pode-se considerar um SGD.

Por fim, existem alguns parâmetros chaves que podem ser alterados na chamada da classe que são cruciais para o funcionamento da rede neural. Eles são: learning rate ('eta' no código), epochs e o número de camadas ocultas, assim como o número de neurônios nela.

# 5. Treinamento do modelo


```python
n = len(df)
split_at = int(n * 0.7)
train = df.iloc[:split_at].copy()
test = df.iloc[split_at:].copy()

print(f"Full shape: {df.shape}")
print(f"Train rows: {len(train)}, Test rows: {len(test)}")
```

    Full shape: (2465, 34)
    Train rows: 1725, Test rows: 740



```python
features = [c for c in train.columns if c != "Target"]

X_train = train[features].values.astype(float)
y_train = train["Target"].values.astype(float)

X_test = test[features].values.astype(float)
y_test = test["Target"].values.astype(float)

model = mlp(
    n_features=X_train.shape[1],
    n_hidden_layers=3,
    n_neurons_per_layer=[64, 32, 16, 1],
    activation="relu",
    loss="mse", # como é uma regressão utilizaremos mse
    optimizer="gd",
    epochs=200,
    eta=1e-4
)

model.train(X_train, y_train)
```

    
    === Inicialização de Pesos e Biases ===
    Camada 1:
    W1 shape (64, 33):
    [[-0.07375728 -0.01234177 -0.06858191 ...  0.09940581 -0.02959893
       0.10308834]
     [-0.0649978  -0.00141824  0.00626712 ... -0.14427376  0.15386409
      -0.06963884]
     [-0.07887087 -0.05496458 -0.18204217 ... -0.10613693  0.01317375
       0.0864927 ]
     ...
     [-0.1148269   0.14581867 -0.00366937 ...  0.02425095 -0.00293177
       0.08459666]
     [-0.19063275 -0.06259786 -0.02439312 ...  0.02750479 -0.13246107
       0.2954073 ]
     [ 0.10322359  0.07390055 -0.03538294 ...  0.1177716   0.02209193
      -0.10829737]]
    b1 shape (64,):
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    
    Camada 2:
    W2 shape (32, 64):
    [[-0.04349707  0.03220502  0.09552407 ... -0.18189484 -0.02694631
      -0.22179883]
     [ 0.16840904 -0.07254595 -0.11931569 ...  0.27033712 -0.03456229
       0.01961908]
     [ 0.03514387  0.056339    0.14698521 ...  0.08951495 -0.14843468
       0.07780176]
     ...
     [-0.01703633  0.04274279  0.00524755 ...  0.07799754  0.08848929
      -0.20819767]
     [ 0.04643976 -0.04597888 -0.13723408 ...  0.17347136 -0.15294575
       0.13161419]
     [-0.15177188 -0.06634908  0.01456536 ...  0.07850615 -0.00500475
       0.02794385]]
    b2 shape (32,):
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0.]
    
    Camada 3:
    W3 shape (16, 32):
    [[ 3.33944095e-02  1.13671688e-01 -8.59073450e-02  1.08972854e-01
      -4.48276520e-02 -7.20058749e-02  1.23725968e-01 -1.64709125e-01
       2.01886915e-01  6.20926831e-02 -9.39526968e-03  2.45061419e-02
       1.55696948e-01 -8.49143194e-02  2.98800890e-02  1.26202938e-01
       7.39188607e-02 -2.49494356e-02 -1.95942924e-01 -1.01543317e-01
       1.42858598e-01  3.17063810e-02 -1.00778552e-01  2.60889795e-01
       4.60774918e-03 -6.25482898e-02  6.56701512e-02 -4.98001408e-02
       1.61696729e-01  7.90584268e-02 -9.57723567e-02 -1.00737857e-01]
     [-7.72170001e-02  7.58457198e-02  7.07874903e-02  3.85769819e-02
       1.23002541e-02 -3.75923201e-02  9.25609597e-02  2.03823407e-03
       3.09605418e-03  9.30878582e-02  1.17593538e-01  1.15905880e-01
      -2.77360351e-02 -2.51220901e-02  8.79446863e-02  9.37835452e-02
       6.44948148e-04  1.40578086e-01  1.73666198e-02 -3.64478793e-02
      -4.00726357e-02  4.27519456e-02  2.23015936e-01 -5.24425581e-03
       8.39940080e-03 -1.47601688e-02  8.16800125e-02 -1.09378668e-01
      -3.20027770e-01  5.22767304e-03 -1.09768577e-02 -1.15983394e-01]
     [ 1.75774890e-01 -1.75025467e-02 -1.11665907e-02  1.75524062e-02
       1.30769140e-01 -2.58768074e-01 -9.93435340e-02  5.65140252e-02
       5.65644007e-02 -2.29234853e-01 -1.42096351e-01 -1.27331642e-01
       7.53757518e-02 -4.90082929e-02 -7.59163895e-03  6.65338887e-02
       1.04194525e-01 -1.14049449e-02  1.40278227e-02 -1.35246819e-01
      -2.58392790e-02  1.29434805e-01  1.55981785e-01 -7.03333530e-02
      -4.89938552e-02  1.85136476e-02  2.95167057e-02 -1.53031792e-01
       1.20338005e-01 -1.02908299e-01 -1.13271104e-01 -5.36195888e-02]
     [-6.06663860e-02  4.95180920e-02  7.01351268e-02 -4.90333305e-03
       1.22437575e-01 -2.25798373e-06 -2.36384440e-02  7.18711569e-02
      -6.97185269e-02  2.18071023e-02  4.36824362e-03 -9.28256994e-02
       5.72023742e-02  8.91571616e-02  2.13441943e-02 -1.05254057e-01
      -1.49161408e-01  2.31853780e-01  1.59363719e-01 -1.51399538e-01
       1.99203263e-01  1.68388836e-01  1.41295846e-02 -1.28566901e-01
       6.21508039e-02 -5.68963281e-02 -1.16832205e-01 -4.46173019e-02
      -3.23211491e-04  2.11473516e-02 -2.61967418e-02 -1.36184163e-01]
     [ 1.37855773e-02  7.05487938e-02  1.98905147e-02  5.48865316e-02
      -1.50421832e-01 -1.57969859e-01 -3.97306170e-02  5.82349782e-04
       1.23133335e-01 -1.97424659e-01  1.36609094e-01 -3.82745168e-02
      -1.13572568e-01 -8.31828647e-02  2.04278385e-02  6.57211069e-02
      -9.20404584e-02  9.52923074e-02 -1.25988903e-01 -7.38406968e-02
      -1.17705663e-01  1.94369498e-02 -1.00964172e-01  1.05298936e-01
      -1.53080942e-01 -1.00385451e-01  7.73343796e-02 -1.01529446e-02
      -1.53778683e-01 -8.77329017e-02 -2.11437759e-01 -1.54612147e-01]
     [ 1.27264557e-01 -8.56002445e-02 -1.28002230e-01  1.66171636e-01
       5.62144834e-02 -4.79794325e-02 -1.16211322e-01  1.19970152e-01
       2.41130379e-02  5.59364584e-02 -2.18066449e-02  5.24481522e-02
       4.80572654e-03  1.59081518e-01 -1.56079078e-02 -2.07544478e-01
       9.17257520e-02 -1.57654818e-01  5.00705796e-02 -1.55318078e-01
      -8.07559913e-02  8.37703899e-02  1.58499802e-01 -4.68866114e-02
       8.97096601e-02 -1.25298977e-01 -1.01651031e-01 -8.66643663e-02
       1.60114630e-01  6.96605552e-02 -1.06004358e-02 -9.77087448e-02]
     [ 1.06345699e-01  5.36075567e-02  4.66119071e-02  1.06792270e-01
      -8.14629769e-02 -3.16252853e-02 -4.41627099e-03 -1.39353401e-01
      -2.37388546e-01  2.23167940e-02 -1.44219380e-02  9.02604040e-03
      -1.49001913e-01  8.23255451e-02  1.08834801e-02  1.01338247e-01
      -1.63203229e-01 -3.46030968e-02  9.73463822e-02 -9.43123612e-02
      -1.40656293e-02 -3.44154945e-03 -1.01245855e-01 -6.91992057e-02
      -2.06294874e-01  1.84277307e-02 -1.28935737e-02 -8.74839492e-02
       1.02317513e-01 -1.09116775e-01 -1.05472479e-01 -4.83874422e-02]
     [ 7.66682040e-02 -8.13641157e-02 -4.25694404e-02  2.02888859e-02
      -8.72235277e-02  3.57552919e-03 -6.15742009e-02  1.16950698e-01
       8.42180030e-02 -1.49595671e-01  2.03449099e-01 -9.87365822e-02
       5.64203470e-02 -6.39442470e-02  6.98217926e-02 -1.30862431e-01
      -3.84893328e-02 -8.62238100e-02 -1.19575115e-01  6.79042372e-02
       6.64421298e-02  1.64088499e-01  1.65096680e-01  1.62246726e-01
      -8.24809000e-03 -1.11670554e-01  3.71896037e-02  1.91721503e-01
       1.51638481e-01  1.46596742e-02 -7.79922528e-02  1.30218038e-02]
     [-2.92069673e-01 -2.00363408e-01 -2.35155089e-02  1.18925579e-01
      -1.04669204e-01 -3.13592609e-02 -9.39444417e-02 -5.73161895e-02
       2.72725542e-03  4.22521714e-02  2.81980681e-02  4.80868665e-02
       1.96413423e-01 -5.79284991e-02  9.69286823e-02  1.61845150e-02
       7.48309433e-02  6.96074872e-02 -4.88607284e-02  1.81151283e-02
       1.68380294e-01 -9.32401692e-02 -7.14278295e-02 -6.95252603e-02
       1.61690504e-02  1.53449935e-02 -1.67061217e-01 -1.19774547e-01
      -2.50085166e-02 -1.35444495e-02  2.23429550e-02  5.38720087e-02]
     [ 1.13031413e-01 -2.11037942e-01  3.52073861e-02 -5.21843052e-02
      -2.40192073e-02  1.62538913e-02 -8.97131737e-02 -1.06050046e-01
      -8.29019301e-02 -4.41397287e-03  4.41805631e-02 -7.52829759e-02
       1.47045253e-01 -1.02037568e-01  1.22104937e-02  2.70304865e-02
      -2.36246381e-03  2.05182921e-01  2.78370668e-02  6.69342297e-03
       4.52825135e-03 -6.62627404e-02 -9.81783353e-02 -2.22076956e-01
       3.85487634e-02  6.24339003e-02 -1.69549765e-01 -1.26011809e-01
       1.12069315e-01 -6.96306023e-02 -9.53077530e-02 -2.35303887e-02]
     [ 5.04658864e-02  1.24002214e-02  2.53093698e-03 -5.97746346e-03
      -5.60577710e-02  2.50535027e-02  2.77402594e-04 -3.14055120e-02
       7.95188075e-02  1.34441753e-02 -6.55913609e-02  2.72504803e-02
       3.35082217e-02  3.69433834e-02 -4.05034148e-02  5.42351433e-02
      -4.02294317e-02  3.34600958e-02  9.19990479e-03  4.54330511e-03
       1.06476178e-01  2.29680429e-01 -9.24902799e-02 -7.57334162e-02
      -5.51514949e-02 -3.58925200e-02  3.11003366e-02  5.45626898e-03
      -1.39565256e-01 -3.19107370e-02 -2.37370843e-01 -1.61998313e-02]
     [ 3.67359477e-02  3.48684922e-02 -1.45856890e-01  4.38677611e-03
       1.02308586e-01  6.57929956e-02 -9.78933816e-02  6.26872890e-02
       7.87881366e-02 -1.53707112e-01  1.71006486e-01  1.22103014e-01
       5.93895211e-02 -1.26020089e-01 -1.25355658e-01  1.45085462e-01
       9.25737861e-02  3.54071772e-03  1.00919649e-01 -1.51328122e-03
       6.50129702e-02 -1.36783437e-01  6.20825438e-02 -9.36726918e-02
      -8.76163015e-02  8.80026055e-03  2.09111122e-02 -2.10150296e-01
       2.77917450e-01 -6.75687332e-02 -5.37915796e-02 -8.39711268e-02]
     [ 7.79631054e-02  8.32000874e-02 -1.75906214e-02  2.62166530e-02
      -1.23334468e-01  6.10172969e-02 -7.20623239e-02 -5.52090131e-02
       6.67463575e-02  2.08562304e-01 -9.29923395e-02  1.55259212e-01
       2.13274400e-01 -2.46983410e-02 -1.64860334e-01 -1.71189568e-01
      -1.68555246e-02  8.09566383e-02 -6.30219208e-02 -6.97897573e-02
       6.31223179e-02  2.28929890e-02 -1.03980706e-01 -1.63959137e-01
      -9.93624427e-02  2.23617345e-02 -1.58508433e-01 -1.03791227e-03
      -6.49050392e-02  2.09731007e-01  6.20474583e-02 -8.18087522e-02]
     [-1.82188138e-01 -5.00853130e-02 -5.99369829e-02  1.31086561e-02
       2.40874603e-02  1.01763435e-01  5.88662161e-02  7.94872229e-02
      -1.02041127e-02  6.53882319e-02  5.39405277e-04  6.08866996e-02
       6.96747594e-02 -4.89248334e-02 -1.04262297e-01  3.47843760e-02
       6.98031867e-03 -1.07127364e-01 -1.56749047e-01  2.21056868e-01
       7.14521866e-03  8.99428117e-02  7.38112760e-03  2.61064883e-02
      -2.28736668e-01  1.57561701e-01  4.90654001e-02  1.07407611e-02
      -3.54449489e-02 -9.83575013e-02 -7.01182393e-03 -3.40289546e-02]
     [ 1.19472654e-01 -4.56421711e-02 -6.76560646e-03 -6.66093168e-02
       5.08335926e-03 -1.00388154e-01 -8.17138624e-02  5.66536402e-03
       1.49484895e-01  2.24405684e-01 -2.43915604e-01 -1.96642186e-01
      -2.83635648e-01 -7.27869964e-02 -6.80440157e-03  4.51631977e-02
      -2.04109963e-02  4.18941665e-02  4.17631235e-02 -1.14426127e-01
       6.51706514e-02  8.36777681e-02  5.07929949e-02 -1.00298639e-01
      -5.14164288e-03 -8.75110088e-02 -1.19771825e-01 -9.54633071e-02
       1.63992120e-01 -1.84353827e-02  1.51822276e-01 -4.84434304e-02]
     [-1.78747210e-02  8.76228808e-02  1.50878105e-01 -6.16588420e-03
       9.25873764e-02 -1.12521875e-01 -1.27380378e-01 -6.07768071e-02
       1.08919650e-01  3.15320917e-02  1.33466175e-01  5.76017934e-02
      -1.38195482e-01  1.87901072e-02 -3.15243061e-02 -1.20970926e-01
       2.78137191e-02  3.42832392e-02  1.18937312e-01  9.59696231e-02
       8.80954735e-02 -1.02197913e-01  9.35910542e-02  1.03574289e-01
      -7.63416035e-02  9.58560024e-02 -1.39874543e-01  2.14288550e-02
       1.21033578e-01  9.01645373e-02 -1.53204950e-01  8.82586456e-02]]
    b3 shape (16,):
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    
    Camada 4:
    W4 shape (1, 16):
    [[-0.00887804 -0.09834723  0.14028586 -0.07073522  0.02417115 -0.001273
       0.06745209  0.14788874 -0.07538337 -0.06816405  0.09116095  0.06282345
      -0.06226588 -0.03753591 -0.26239777  0.12356502]]
    b4 shape (1,):
    [0.]
    
    Epoch 0, Loss: 3618.041790, Train RMSE: 85.124483
    Epoch 10, Loss: 2354.617543, Train RMSE: 48.802098
    Epoch 20, Loss: 2372.061626, Train RMSE: 48.968207
    Epoch 30, Loss: 2372.909946, Train RMSE: 48.975558
    Epoch 40, Loss: 2372.937164, Train RMSE: 48.975794
    Epoch 50, Loss: 2372.938028, Train RMSE: 48.975801
    Treinamento encerrado no epoch 51 (convergência detectada).



```python
preds_test = model.test(X_test)
```

# 6. Estratégia de treino e teste
Para o processo de modelagem, o conjunto de dados foi dividido em 70% para treino e 30% para teste.

Optou-se por um treinamento estocástico, no qual os pesos da rede são atualizados a cada amostra (em vez de após todo o conjunto de dados). Essa abordagem foi escolhida por acelerar a convergência e permitir que o modelo explore o espaço de soluções de forma mais dinâmica, reduzindo o risco de ficar preso em mínimos locais.

Para evitar overfitting, foi implementado um mecanismo de early stopping.
A lógica consiste em monitorar o loss médio em uma janela deslizante de epochs (parâmetro ajustável) e verificar se a diferença entre o loss atual e a média do loss nessa janela é menor que um limite (threshold) de parada.
Quando essa condição é satisfeita, entende-se que o modelo atingiu convergência, e o treinamento é interrompido antecipadamente, prevenindo o ajuste excessivo aos dados de treino e melhorando a capacidade de generalização.

# 7. Curva de erro e visualização


```python
report = model.evaluate(X_test, y_test, plot_confusion=False, plot_roc=False, preds=preds_test)
```

    
    Regression metrics (model vs. mean baseline):
    Metric               Model      Baseline
    ----------------------------------------
    MAE             115.558561     26.897059
    MAPE (%)         57.905644     14.412587
    MSE           14374.127041   1020.338675
    RMSE            119.892148     31.942741
    R2              -13.087604      0.000000
    
    === Pesos e Biases do Modelo ===
    
    Camada 1:
      Pesos W1 (shape (64, 33)):
    [[-0.14761669 -0.08605595 -0.14341255 ...  0.07323756 -0.13721254
       0.0298664 ]
     [-0.06511377 -0.00109304  0.0071624  ... -0.14972238  0.15166351
      -0.05035075]
     [-0.0680096  -0.04379908 -0.17075397 ... -0.10853388  0.03554515
       0.10479415]
     ...
     [-0.12447155  0.13607814 -0.0124113  ... -0.02101909 -0.12513566
      -0.01549921]
     [-0.29867799 -0.17052402 -0.13167242 ...  0.02518173 -0.25437548
       0.23854131]
     [ 0.09856172  0.06879592 -0.0402705  ...  0.12015683  0.01190544
      -0.09469006]]
      Biases b1 (shape (64,)):
    [ 0.02438693 -0.02645427 -0.01353966 -0.11836173 -0.28734522 -0.00563088
     -0.21293918  0.41206632 -0.05597868 -0.11097652 -0.01740126 -0.08038929
      0.02050711 -0.07971225 -0.06483242 -0.07192842 -0.28176881 -0.03847096
      0.00634015  0.03133328 -0.0279558  -0.00317982 -0.23968182 -0.02222732
      0.01354029  0.00126747  0.51176482  0.03421154 -0.11133629 -0.0285405
     -0.04525796 -0.21066117 -0.02010222 -0.04463554 -0.09443288  0.34239399
     -0.18972995 -0.01274602  0.02285582 -0.0150045  -0.01293346 -0.02129619
      0.005671   -0.02224317 -0.01880599 -0.01091744  0.06590349  0.29010453
     -0.05863941  0.09746505 -0.10927989 -0.07194447  0.03997355 -0.02028761
     -0.00617349 -0.01316064 -0.15526599  0.81975633 -0.08216532 -0.0427027
     -0.03976054 -0.09389672  0.09141876 -0.01287721]
    
    Camada 2:
      Pesos W2 (shape (32, 64)):
    [[-0.24897238  0.04767612  0.10160991 ... -0.2368596   0.25206535
      -0.24100288]
     [ 0.08338775 -0.04076193 -0.10948062 ...  0.22288544  0.00700335
       0.03554049]
     [-0.05165429  0.06677419  0.14996807 ...  0.06573969  0.00825065
       0.07423783]
     ...
     [-0.18477174  0.07152586  0.01640215 ... -0.00419478 -0.02235338
      -0.18174787]
     [ 0.03354841 -0.04769328 -0.13866027 ...  0.1622688  -0.17961952
       0.13015774]
     [-0.1522655  -0.06836642  0.01433697 ...  0.07825661 -0.01709234
       0.02765171]]
      Biases b2 (shape (32,)):
    [ 5.86216896e-01  4.35081488e-01  2.83204519e-01  3.63898301e-01
      1.18264962e-01  1.49628497e-01 -3.01983720e-02  1.45758815e-02
      7.20194889e-02 -1.81053632e-02  3.55858477e-03  1.23065546e-01
      8.28709307e-03  2.28866968e-03  4.39577212e-02  1.34273784e-01
      2.40035548e-01  8.50182952e-02  1.46104784e-04 -2.69669074e-05
      7.99498299e-01 -8.79239663e-04  7.32308682e-01  9.43599149e-01
      7.56729953e-02 -8.34230584e-03  5.68814186e-01  2.03052539e-02
      4.22430870e-01  4.65010145e-01 -5.65595393e-02 -2.37081421e-02]
    
    Camada 3:
      Pesos W3 (shape (16, 32)):
    [[-2.91086744e-02 -2.13231157e-01 -1.07114487e-01 -2.81012639e-01
      -1.75112353e-01 -3.06722940e-01  1.23735513e-01 -2.02565429e-01
      -3.22358461e-02  6.98570148e-02 -9.39526968e-03 -1.16027194e-01
       9.06004266e-02 -9.28016314e-02  3.14382744e-02  1.13336839e-01
      -1.81747767e-01 -1.47300312e-02 -1.95942924e-01 -1.02419811e-01
      -6.41733735e-01  1.91543632e-02 -6.88391727e-01 -6.78131054e-01
      -1.45372788e-01 -6.04517569e-02 -2.55099071e-01 -1.04380675e-01
      -4.67026237e-01 -4.32251504e-01 -8.35689733e-02 -1.00568099e-01]
     [-1.75557291e-03 -6.47212040e-02 -2.05575790e-02 -2.01386398e-02
       1.48929772e-02 -3.65774654e-02  9.06559588e-02  3.32821917e-03
       3.09605418e-03  9.49529923e-02  1.17593538e-01  1.76504396e-02
      -2.77360351e-02 -1.69724755e-02 -1.73952907e-02 -3.19157206e-02
       3.49918866e-03 -6.55476625e-02  1.73666198e-02 -3.83089096e-02
      -1.48375140e-02  4.16381961e-02  4.02019194e-03 -5.21039039e-02
       1.74084827e-03  3.98316751e-03 -6.18967431e-01 -1.09378668e-01
      -2.06581531e-01  7.44292123e-02 -2.26329186e-02 -8.85422395e-02]
     [ 4.61436880e-02 -5.89707850e-02 -6.68616714e-02 -1.11649441e-02
       1.20761805e-01 -2.77910456e-01 -1.02494123e-01  5.20744385e-02
       3.56104830e-02 -2.29225571e-01 -1.42096351e-01 -1.40173865e-01
       6.93855759e-02 -5.07596535e-02 -7.99586486e-03  2.76695265e-02
       8.50588365e-02 -1.14328014e-02  1.40278227e-02 -1.35468935e-01
      -8.65638476e-02  1.29957622e-01  1.11689327e-01 -1.36120169e-01
      -6.33588522e-02  1.90069473e-02 -3.17629535e-02 -1.58102934e-01
       7.98052279e-02 -1.37835173e-01 -1.13267526e-01 -5.33088521e-02]
     [-6.57185099e-02  2.59604887e-02  6.56573746e-02 -1.65985128e-02
       1.12658768e-01 -1.37428720e-02 -2.56843911e-02  7.15157608e-02
      -6.99702486e-02  2.08303289e-02  4.36824362e-03 -9.72001841e-02
       5.72023742e-02  8.91220885e-02  1.95004302e-02 -1.10984485e-01
      -1.58377796e-01  2.25256360e-01  1.59363719e-01 -1.51503524e-01
       1.63945923e-01  1.67324715e-01 -1.22819015e-02 -1.57628182e-01
       4.98095745e-02 -5.69392220e-02 -1.29110887e-01 -4.46173019e-02
      -7.78696196e-03  9.92740339e-04 -3.36733417e-02 -1.36291181e-01]
     [-1.91676824e-01  9.71958159e-02 -6.68462552e-02  7.66977466e-02
      -1.39179700e-01 -1.46730006e-01 -3.97434618e-02  5.82349782e-04
       1.23008564e-01 -1.97718163e-01  1.36573004e-01 -3.87570141e-02
      -1.13572568e-01 -8.25537180e-02  2.30949806e-02 -3.16261466e-02
      -7.82157347e-02  9.31139073e-02 -1.26068012e-01 -7.36773058e-02
      -5.96317767e-02  1.93586523e-02 -5.02845395e-02  1.65931541e-01
      -1.45357843e-01 -1.00412735e-01 -8.66078208e-02 -1.01529446e-02
      -1.27244180e-01 -5.92280073e-02 -2.11415202e-01 -1.52357458e-01]
     [ 1.04901861e-01 -1.73609969e-01 -1.32652431e-01  6.21711821e-02
       2.06235960e-02 -1.12711403e-01 -1.16864430e-01  1.10301735e-01
      -3.50399986e-02  5.78322853e-02 -2.18066449e-02  1.51221193e-02
      -1.16286995e-02  1.57077039e-01 -1.52536510e-02 -2.06746104e-01
       2.27483667e-02 -1.55158542e-01  5.00705796e-02 -1.55546696e-01
      -2.92033876e-01  8.00305242e-02  7.38767419e-06 -2.98446079e-01
       4.84344781e-02 -1.24763807e-01 -1.16819234e-01 -1.00563280e-01
      -4.77643003e-03 -6.82282617e-02 -8.25722059e-03 -9.76663009e-02]
     [-1.62517097e+00  1.80639500e-01 -7.05346516e-01  1.64245533e-01
      -7.37156262e-02 -1.07664323e-02 -6.21776057e-03 -1.45357821e-01
      -2.41569047e-01  8.15505561e-03 -1.82314578e-02 -2.64113019e-03
      -1.49001913e-01  8.24100470e-02  1.79377564e-02 -7.30099468e-01
      -1.40185104e-01 -7.63348882e-02  9.73463822e-02 -9.33549681e-02
       1.33683512e-01 -8.70166754e-03  8.69117754e-03  5.59986776e-02
      -2.01565441e-01  1.09928518e-02 -1.77127820e-01 -8.87910800e-02
       1.45414437e-01 -4.35243750e-02 -1.05253203e-01 -2.75838060e-02]
     [ 5.07045728e-02 -4.83358868e-01 -4.13136432e-02 -4.72953571e-01
      -2.42670062e-01 -2.69985626e-01 -5.54372168e-02  6.56380836e-02
      -2.34843587e-01 -1.37766126e-01  2.03449099e-01 -2.77035739e-01
      -3.24972576e-02 -7.44254800e-02  7.42929732e-02 -1.08220047e-01
      -3.55048529e-01 -7.13051305e-02 -1.19575115e-01  6.66897220e-02
      -9.09716596e-01  1.50443853e-01 -5.66028784e-01 -1.02624978e+00
      -1.79085855e-01 -1.08710906e-01 -3.53136320e-01  1.17635417e-01
      -6.80686368e-01 -6.17383063e-01 -4.24303166e-02  1.32641868e-02]
     [-2.92106320e-01 -2.03748125e-01 -2.38849505e-02  1.13343995e-01
      -1.06265643e-01 -3.69718615e-02 -9.45611456e-02 -5.73161895e-02
       2.72725542e-03  4.16813517e-02  2.81980681e-02  4.62713040e-02
       1.96413423e-01 -5.79284991e-02  9.61652128e-02  1.41239888e-02
       7.03786973e-02  6.92947328e-02 -4.88607284e-02  1.81151283e-02
       1.57866691e-01 -9.34967268e-02 -7.77283990e-02 -7.83035157e-02
       1.15916147e-02  1.53449935e-02 -1.67417595e-01 -1.19774547e-01
      -2.70176580e-02 -2.05782012e-02  1.86977261e-02  5.38720087e-02]
     [ 6.64049484e-02 -2.10497235e-01  1.55944973e-02 -5.21522417e-02
      -2.40193251e-02  1.62538913e-02 -8.97131737e-02 -1.06050046e-01
      -8.29019301e-02 -4.95127074e-03  4.41805631e-02 -7.52829759e-02
       1.47045253e-01 -1.02037568e-01  1.22112958e-02  9.55768236e-03
      -2.36201555e-03  2.03186162e-01  2.78370668e-02  6.69342297e-03
       4.67351768e-03 -6.62627404e-02 -9.81015518e-02 -2.22049989e-01
       3.85487634e-02  6.24339003e-02 -1.71139472e-01 -1.26011809e-01
       1.12070501e-01 -6.96149796e-02 -9.53056946e-02 -2.35303887e-02]
     [-1.23993561e+00  1.24677693e-01 -5.54785683e-01  5.30216747e-02
      -3.78826896e-02  5.20801731e-02 -9.11281185e-04 -3.85213951e-02
       7.66694780e-02  3.81373907e-03 -6.81145718e-02  1.91597640e-02
       3.35082217e-02  3.29492743e-02 -3.47932835e-02 -5.66775864e-01
      -7.48467659e-03  7.52419979e-03  9.15173107e-03  2.94603144e-03
       2.60235316e-01  2.19387735e-01  3.34851652e-02  7.11751637e-02
      -4.18115769e-02 -4.07101960e-02 -2.10911345e-01  1.22341068e-03
      -8.32453472e-02  4.46259323e-02 -2.37786959e-01 -1.48139986e-04]
     [-1.59098408e-03 -1.34446083e-01 -1.56993149e-01 -2.02737361e-01
       3.64656263e-02 -5.07101587e-02 -9.61731363e-02  4.14753596e-02
      -5.22836126e-02 -1.48819133e-01  1.71006486e-01  4.74553374e-02
       2.28838032e-02 -1.30331180e-01 -1.23437484e-01  1.37955215e-01
      -4.06132372e-02  9.51199915e-03  1.00919649e-01 -2.00986453e-03
      -3.46219416e-01 -1.42893520e-01 -2.46324385e-01 -5.93930182e-01
      -1.60005194e-01  9.98437822e-03 -1.53587577e-01 -2.40679999e-01
      -6.97530776e-02 -3.34027173e-01 -3.91174268e-02 -8.38734385e-02]
     [ 4.50438093e-02  6.75978048e-02 -3.18839836e-02  2.26628274e-02
      -1.24946087e-01  5.68808133e-02 -7.35238852e-02 -5.52625754e-02
       6.65327774e-02  2.07425481e-01 -9.29923395e-02  1.53367014e-01
       2.13274400e-01 -2.50911120e-02 -1.65328804e-01 -1.76668995e-01
      -1.96763769e-02  7.99927029e-02 -6.30219208e-02 -6.98136946e-02
       5.29737483e-02  2.21939273e-02 -1.08963616e-01 -1.69809590e-01
      -1.03333548e-01  2.23617345e-02 -1.62948586e-01 -1.03791227e-03
      -6.54183123e-02  2.04361067e-01  5.85381045e-02 -8.18518751e-02]
     [-1.81900348e-01 -4.71175302e-02 -5.95325014e-02  1.51991924e-02
       2.52073488e-02  1.02754866e-01  5.88662161e-02  7.92806673e-02
      -1.02041127e-02  6.53226350e-02  5.39405277e-04  6.08008913e-02
       6.96747594e-02 -4.92804534e-02 -1.04008983e-01  3.49673367e-02
       8.29293142e-03 -1.07217682e-01 -1.56749047e-01  2.21007237e-01
       1.24539188e-02  8.95287567e-02  1.21013128e-02  3.17486242e-02
      -2.27929507e-01  1.57481149e-01 -6.18991764e-03  1.06677650e-02
      -3.29541627e-02 -9.58970728e-02 -7.02340753e-03 -3.40289546e-02]
     [ 1.64588287e-02 -5.14831852e-02 -5.09997685e-02 -6.96412130e-02
       7.40000948e-04 -1.02891092e-01 -8.23490238e-02  5.66536402e-03
       1.48301920e-01  2.21928469e-01 -2.44483775e-01 -1.98474120e-01
      -2.83635648e-01 -7.30421296e-02 -8.41047241e-03  1.28851701e-02
      -2.48814740e-02  3.97292568e-02  4.17631235e-02 -1.15057789e-01
       4.99865230e-02  8.28724695e-02  3.79142583e-02 -1.11076657e-01
      -8.18871669e-03 -8.75110088e-02 -1.22592140e-01 -9.56417575e-02
       1.58967788e-01 -2.51362044e-02  1.49653799e-01 -4.88107612e-02]
     [-4.34274879e-02 -2.71953888e-01  1.47666629e-01 -4.47161762e-01
      -4.69422065e-02 -3.58063045e-01 -1.21653040e-01 -1.06276289e-01
      -1.74792780e-01  4.18418157e-02  1.33466175e-01 -1.01650992e-01
      -2.17222162e-01  9.48031695e-03 -2.79376791e-02 -1.02171716e-01
      -2.55853163e-01  4.76950029e-02  1.18937312e-01  9.48767283e-02
      -7.85017636e-01 -1.14438937e-01 -5.59969386e-01 -9.57331130e-01
      -2.30671349e-01  9.84961253e-02 -2.85358133e-01 -4.43486831e-02
      -6.20373566e-01 -4.75445492e-01 -1.23763996e-01  8.84766404e-02]]
      Biases b3 (shape (16,)):
    [-0.0639368   0.34840409 -0.0497138  -0.04719775 -0.00489687 -0.03902983
     -0.03565178 -0.04648489 -0.01504132 -0.02560985 -0.02372781 -0.04592194
     -0.02876675 -0.00172989 -0.08098173 -0.03730276]
    
    Camada 4:
      Pesos W4 (shape (1, 16)):
    [[-1.14326274 -1.50918791 -0.2432558  -0.04544077 -0.14346639 -0.28887588
      -0.85976506 -1.46844213 -0.06867526 -0.17806586 -0.54361284 -0.60762965
      -0.03587944 -0.00853893 -0.24547125 -1.30926224]]
      Biases b4 (shape (1,)):
    [80.05162071]



    
![png](regression_files/regression_26_1.png)
    


# 8. Avaliação do modelo

O modelo de **VAE** apresentou desempenho **inferior ao baseline da média** em todas as métricas de regressão analisadas.

Em termos quantitativos, o modelo obteve um **MAE de aproximadamente 115,6** e **RMSE de 119,9**, enquanto o baseline — que simplesmente prevê a média do alvo — apresentou valores bem menores (**MAE ≈ 26,9** e **RMSE ≈ 31,9**). O **MAPE** reforça essa diferença, com **57,9%** para o modelo contra **14,4%** para o baseline, indicando uma imprecisão relativa cerca de quatro vezes maior. Além disso, o **MSE** elevado (14 374 vs. 1 020) e o **R² negativo (-13,08)** evidenciam que o modelo não conseguiu capturar a variabilidade dos dados, performando pior do que uma simples predição constante (baseline com R² = 0).

Esses resultados sugerem que o **VAE não conseguiu aprender adequadamente o padrão da variável-alvo**, possivelmente devido a problemas de **normalização**, **escala das variáveis**, **hiperparâmetros** (como taxa de aprendizado e número de épocas) ou até mesmo **excesso de complexidade da arquitetura**, o que pode ter levado a **overfitting**.

Durante o treinamento, foi implementada uma **estratégia de early stopping** com o objetivo de mitigar o overfitting. Essa técnica monitorava o **loss médio em uma janela deslizante de epochs**, interrompendo o treinamento quando a **diferença entre o loss atual e a média dessa janela** ficava abaixo de um **limiar (threshold)** definido. No entanto, essa estratégia **não foi eficaz em promover a generalização**, indicando que o modelo convergiu para uma solução local que não se traduz bem para novos dados.

Por fim, o **baseline de referência** consistiu em prever a **média da variável alvo** para todas as amostras, fornecendo uma base simples, porém sólida, de comparação.
O fato de o modelo VAE não superar esse baseline demonstra que o aprendizado foi ineficiente e que **revisões na arquitetura, nos dados de entrada e na função de perda** são necessárias para alcançar resultados mais satisfatórios.




