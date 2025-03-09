import pandas as pd


def dataframe_coeficientes(coefs, colunas):
    return pd.DataFrame(data=coefs, index=colunas, columns=["coeficiente"]).sort_values(
        by="coeficiente"
    )
"""Cria e organiza um DataFrame com coeficientes de um modelo de regressão.

    Esta função constrói um DataFrame a partir de uma lista ou array de coeficientes,
    associando cada coeficiente a uma coluna (feature) especificada, e ordena os coeficientes
    em ordem crescente.

    Args:
        coefs (array-like): Valores dos coeficientes do modelo, geralmente obtidos de um
            atributo como .coef_ de um regressor treinado (ex.: LinearRegression).
        colunas (array-like): Nomes das colunas (features) correspondentes aos coeficientes,
            com o mesmo comprimento que coefs.

    Returns:
        pd.DataFrame: DataFrame com uma coluna 'coeficiente' contendo os valores dos coeficientes,
            indexado pelos nomes das colunas (features), ordenado pelo valor do coeficiente
            em ordem crescente.

    Examples:
        >>> import pandas as pd
        >>> coefs = [1.5, -0.8, 0.2]
        >>> colunas = ['feature1', 'feature2', 'feature3']
        >>> df = dataframe_coeficientes(coefs, colunas)
        >>> print(df)
                 coeficiente
        feature2        -0.8
        feature3         0.2
        feature1         1.5
    """