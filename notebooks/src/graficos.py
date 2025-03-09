import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import EngFormatter
from sklearn.metrics import PredictionErrorDisplay

from .models import RANDOM_STATE

sns.set_theme(palette="bright")

PALETTE = "coolwarm"
SCATTER_ALPHA = 0.2


def plot_coeficientes(df_coefs, tituto="Coeficientes"):
    df_coefs.plot.barh()
    plt.title(tituto)
    plt.axvline(x=0, color=".5")
    plt.xlabel("Coeficientes")
    plt.gca().get_legend().remove()
    plt.show()
"""Plota os coeficientes de um modelo de regressão em um gráfico de barras horizontais.

    Esta função cria um gráfico de barras horizontais para visualizar os coeficientes de um
    modelo de regressão armazenados em um DataFrame, com uma linha vertical em x=0 para
    destacar coeficientes positivos e negativos. O título do gráfico é personalizável.

    Args:
        df_coefs (pd.DataFrame): DataFrame contendo os coeficientes do modelo, onde as linhas
            representam variáveis (features) e há pelo menos uma coluna com os valores dos
            coeficientes (ex.: 'coef'). A primeira coluna numérica é usada para o plot.
        tituto (str, optional): Título do gráfico. Defaults to "Coeficientes".

    Examples:
        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt
        >>> df = pd.DataFrame({'feature': ['A', 'B', 'C'], 'coef': [1.5, -0.8, 0.2]})
        >>> df.set_index('feature', inplace=True)
        >>> plot_coeficientes(df, tituto="Coeficientes do Modelo")
        # Gera um gráfico de barras horizontais com título "Coeficientes do Modelo",
        # barras para 1.5, -0.8 e 0.2, e uma linha vertical em x=0.
    """

def plot_residuos(y_true, y_pred):
    residuos = y_true - y_pred

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    sns.histplot(residuos, kde=True, ax=axs[0])

    error_display_01 = PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="residual_vs_predicted", ax=axs[1]
    )

    error_display_02 = PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="actual_vs_predicted", ax=axs[2]
    )

    plt.tight_layout()

    plt.show()
"""Plota gráficos de resíduos para análise de um modelo de regressão.

    Esta função gera três gráficos em uma única figura para avaliar os resíduos de um modelo
    de regressão: (1) um histograma dos resíduos com curva de densidade (KDE), (2) um gráfico
    de resíduos versus valores previstos, e (3) um gráfico de valores reais versus previstos.
    Esses gráficos ajudam a diagnosticar a qualidade das previsões do modelo.

    Args:
        y_true (array-like): Valores reais da variável alvo (target), geralmente uma Série ou array.
        y_pred (array-like): Valores previstos pelo modelo, com o mesmo tamanho que y_true.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> y_true = np.array([1, 2, 3, 4, 5])
        >>> y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        >>> plot_residuos(y_true, y_pred)
        # Gera uma figura com três subplots:
        # 1. Histograma dos resíduos (ex.: [-0.1, 0.1, -0.2, 0.2, -0.1]) com KDE.
        # 2. Gráfico de resíduos vs. previstos (scatter com linha em y=0).
        # 3. Gráfico de reais vs. previstos (scatter com linha de igualdade).
    """

def plot_residuos_estimador(estimator, X, y, eng_formatter=False, fracao_amostra=0.25):

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    error_display_01 = PredictionErrorDisplay.from_estimator(
        estimator,
        X,
        y,
        kind="residual_vs_predicted",
        ax=axs[1],
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},
        subsample=fracao_amostra,
    )

    error_display_02 = PredictionErrorDisplay.from_estimator(
        estimator,
        X,
        y,
        kind="actual_vs_predicted",
        ax=axs[2],
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},
        subsample=fracao_amostra,
    )

    residuos = error_display_01.y_true - error_display_01.y_pred

    sns.histplot(residuos, kde=True, ax=axs[0])

    if eng_formatter:
        for ax in axs:
            ax.yaxis.set_major_formatter(EngFormatter())
            ax.xaxis.set_major_formatter(EngFormatter())

    plt.tight_layout()

    plt.show()
"""Plota gráficos de resíduos para análise de um estimador de regressão.

    Esta função gera três gráficos em uma única figura para avaliar os resíduos de um estimador
    de regressão treinado: (1) um histograma dos resíduos com curva de densidade (KDE), 
    (2) um gráfico de resíduos versus valores previstos, e (3) um gráfico de valores reais 
    versus previstos. Os gráficos são baseados em uma subamostra dos dados, definida pela 
    fração especificada.

    Args:
        estimator (estimator): Modelo de regressão treinado (ex.: LinearRegression, 
            RandomForestRegressor) que implementa .predict().
        X (array-like): Dados de entrada (features) para o modelo, geralmente um DataFrame ou array.
        y (array-like): Valores reais da variável alvo (target), geralmente uma Série ou array.
        eng_formatter (bool, optional): Se True, aplica formatação de engenharia (ex.: '1.2k' 
            para 1200) aos eixos x e y. Defaults to False.
        fracao_amostra (float, optional): Fração dos dados a serem usados na subamostragem 
            para os gráficos (valor entre 0 e 1). Defaults to 0.25.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from sklearn.linear_model import LinearRegression
        >>> X = pd.DataFrame({'feature': np.arange(100)})
        >>> y = X['feature'] * 2 + np.random.normal(0, 1, 100)
        >>> estimator = LinearRegression().fit(X, y)
        >>> RANDOM_STATE = 42
        >>> SCATTER_ALPHA = 0.5
        >>> plot_residuos_estimador(estimator, X, y, fracao_amostra=0.5)
        # Gera uma figura com três subplots usando 50% dos dados:
        # 1. Histograma dos resíduos com KDE.
        # 2. Gráfico de resíduos vs. previstos (scatter com alpha=0.5).
        # 3. Gráfico de reais vs. previstos (scatter com alpha=0.5).
    """

def plot_comparar_metricas_modelos(df_resultados):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    comparar_metricas = [
        "time_seconds",
        "test_r2",
        "test_neg_mean_absolute_error",
        "test_neg_root_mean_squared_error",
    ]

    nomes_metricas = [
        "Tempo (s)",
        "R²",
        "MAE",
        "RMSE",
    ]

    for ax, metrica, nome in zip(axs.flatten(), comparar_metricas, nomes_metricas):
        sns.boxplot(
            x="model",
            y=metrica,
            data=df_resultados,
            ax=ax,
            showmeans=True,
        )
        ax.set_title(nome)
        ax.set_ylabel(nome)
        ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()

    plt.show()
"""Plota boxplots para comparar métricas de desempenho entre modelos de regressão.

    Esta função gera uma grade de 2x2 boxplots para visualizar e comparar quatro métricas
    de desempenho (tempo total, R², MAE e RMSE) entre diferentes modelos de regressão.
    Cada boxplot mostra a distribuição das métricas por modelo, com a média destacada.

    Args:
        df_resultados (pd.DataFrame): DataFrame contendo os resultados dos modelos,
            com colunas 'model' (nome do modelo) e as métricas 'time_seconds',
            'test_r2', 'test_neg_mean_absolute_error', 'test_neg_root_mean_squared_error'.
            Geralmente obtido de uma função como organiza_resultados().

    Examples:
        >>> import pandas as pd
        >>> import seaborn as sns
        >>> import matplotlib.pyplot as plt
        >>> df = pd.DataFrame({
        ...     'model': ['Linear', 'Linear', 'Forest', 'Forest'],
        ...     'time_seconds': [0.1, 0.2, 0.5, 0.6],
        ...     'test_r2': [0.8, 0.85, 0.9, 0.88],
        ...     'test_neg_mean_absolute_error': [-1.2, -1.1, -0.9, -0.95],
        ...     'test_neg_root_mean_squared_error': [-1.5, -1.4, -1.2, -1.3]
        ... })
        >>> plot_comparar_metricas_modelos(df)
        # Gera uma figura 2x2 com boxplots:
        # 1. Tempo (s) por modelo.
        # 2. R² por modelo.
        # 3. MAE por modelo (valores negativos).
        # 4. RMSE por modelo (valores negativos).
    """