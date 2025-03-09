import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def construir_pipeline_modelo_regressao(
    regressor, preprocessor=None, target_transformer=None
):
    if preprocessor is not None:
        pipeline = Pipeline([("preprocessor", preprocessor), ("reg", regressor)])
    else:
        pipeline = Pipeline([("reg", regressor)])

    if target_transformer is not None:
        model = TransformedTargetRegressor(
            regressor=pipeline, transformer=target_transformer
        )
    else:
        model = pipeline
    return model

"""Constrói um pipeline de regressão com pré-processamento e transformação de alvo opcionais.

    Esta função cria um modelo de regressão encapsulado em um pipeline do scikit-learn,
    permitindo a inclusão opcional de um pré-processador para os dados de entrada (X)
    e/ou um transformador para a variável alvo (y). Se nenhum pré-processador ou transformador
    for fornecido, retorna apenas o regressor em um pipeline básico.

    Args:
        regressor (estimator): O modelo de regressão a ser utilizado (ex.: LinearRegression,
            RandomForestRegressor).
        preprocessor (transformer, optional): Um objeto de pré-processamento (ex.: ColumnTransformer)
            para transformar os dados de entrada (X) antes da regressão. Se None, nenhum
            pré-processamento é aplicado. Defaults to None.
        target_transformer (transformer, optional): Um objeto de transformação (ex.: StandardScaler)
            para transformar a variável alvo (y) antes do ajuste e reverter após a previsão.
            Se None, nenhuma transformação é aplicada ao alvo. Defaults to None.

    Returns:
        model (Pipeline ou TransformedTargetRegressor): O modelo de regressão encapsulado.
            - Se `target_transformer` for None, retorna um `Pipeline`.
            - Se `target_transformer` for fornecido, retorna um `TransformedTargetRegressor`
              contendo o pipeline.
    """
def treinar_e_validar_modelo_regressao(
    X,
    y,
    regressor,
    preprocessor=None,
    target_transformer=None,
    n_splits=5,
    random_state=RANDOM_STATE,
):

    model = construir_pipeline_modelo_regressao(
        regressor, preprocessor, target_transformer
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring=[
            "r2",
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ],
    )

    return scores
"""Treina e valida um modelo de regressão usando validação cruzada.

    Esta função constrói um pipeline de regressão com um regressor, um pré-processador opcional
    para as features (X) e um transformador opcional para o alvo (y). Em seguida, realiza
    validação cruzada com K-Fold para avaliar o desempenho do modelo usando métricas
    específicas de regressão.

    Args:
        X (array-like): Dados de entrada (features) para o modelo, geralmente um DataFrame ou array.
        y (array-like): Variável alvo (target) a ser prevista, geralmente uma Série ou array.
        regressor (estimator): O modelo de regressão a ser treinado (ex.: LinearRegression,
            RandomForestRegressor).
        preprocessor (transformer, optional): Um objeto de pré-processamento (ex.: ColumnTransformer)
            para transformar os dados de entrada (X) antes da regressão. Se None, nenhum
            pré-processamento é aplicado. Defaults to None.
        target_transformer (transformer, optional): Um objeto de transformação (ex.: StandardScaler)
            para transformar a variável alvo (y) antes do ajuste e reverter após a previsão.
            Se None, nenhuma transformação é aplicada ao alvo. Defaults to None.
        n_splits (int, optional): Número de divisões (folds) para a validação cruzada K-Fold.
            Defaults to 5.
        random_state (int, optional): Semente para a aleatoriedade no embaralhamento dos dados
            na validação cruzada. Deve ser uma constante definida como RANDOM_STATE.

    Returns:
        dict: Dicionário contendo os resultados da validação cruzada para cada métrica:
            - 'test_r2': Coeficiente de determinação (R²) para cada fold.
            - 'test_neg_mean_absolute_error': Erro absoluto médio negativo (MAE) para cada fold.
            - 'test_neg_root_mean_squared_error': Erro quadrático médio negativo (RMSE) para cada fold.
            Além disso, inclui tempos de ajuste e pontuação ('fit_time', 'score_time').
    """

def grid_search_cv_regressor(
    regressor,
    param_grid,
    preprocessor=None,
    target_transformer=None,
    n_splits=5,
    random_state=RANDOM_STATE,
    return_train_score=False,
):
    model = construir_pipeline_modelo_regressao(
        regressor, preprocessor, target_transformer
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        model,
        cv=kf,
        param_grid=param_grid,
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
        refit="neg_root_mean_squared_error",
        n_jobs=-1,
        return_train_score=return_train_score,
        verbose=1,
    )

    return grid_search
"""Realiza busca em grade (grid search) com validação cruzada para um modelo de regressão.

    Esta função constrói um pipeline de regressão com um regressor base, um pré-processador
    opcional para as features (X) e um transformador opcional para o alvo (y). Em seguida,
    usa GridSearchCV do scikit-learn para testar combinações de hiperparâmetros definidas
    em param_grid, avaliando o desempenho com validação cruzada K-Fold e múltiplas métricas.

    Args:
        regressor (estimator): O modelo de regressão base a ser otimizado (ex.: LinearRegression,
            RandomForestRegressor).
        param_grid (dict): Dicionário com os hiperparâmetros a serem testados no formato
            {'nome_parametro': [valores]}. Exemplo: {'reg__n_estimators': [50, 100]}.
        preprocessor (transformer, optional): Um objeto de pré-processamento (ex.: ColumnTransformer)
            para transformar os dados de entrada (X) antes da regressão. Se None, nenhum
            pré-processamento é aplicado. Defaults to None.
        target_transformer (transformer, optional): Um objeto de transformação (ex.: StandardScaler)
            para transformar a variável alvo (y) antes do ajuste e reverter após a previsão.
            Se None, nenhuma transformação é aplicada ao alvo. Defaults to None.
        n_splits (int, optional): Número de divisões (folds) para a validação cruzada K-Fold.
            Defaults to 5.
        random_state (int, optional): Semente para a aleatoriedade no embaralhamento dos dados
            na validação cruzada. Deve ser uma constante definida como RANDOM_STATE.
        return_train_score (bool, optional): Se True, retorna as pontuações de treino além das
            pontuações de teste no resultado do GridSearchCV. Defaults to False.

    Returns:
        GridSearchCV: Objeto GridSearchCV configurado e pronto para ser ajustado aos dados
            com .fit(X, y). Após o ajuste, contém o melhor modelo (best_estimator_) e os
            resultados da busca (cv_results_).
    """

def organiza_resultados(resultados):

    for chave, valor in resultados.items():
        resultados[chave]["time_seconds"] = (
            resultados[chave]["fit_time"] + resultados[chave]["score_time"]
        )

    df_resultados = (
        pd.DataFrame(resultados).T.reset_index().rename(columns={"index": "model"})
    )

    df_resultados_expandido = df_resultados.explode(
        df_resultados.columns[1:].to_list()
    ).reset_index(drop=True)

    try:
        df_resultados_expandido = df_resultados_expandido.apply(pd.to_numeric)
    except ValueError:
        pass

    return df_resultados_expandido
"""Organiza os resultados de validação cruzada de modelos em um DataFrame expandido.

    Esta função processa um dicionário de resultados de validação cruzada, adiciona uma métrica
    de tempo total (soma de fit_time e score_time), converte os dados em um DataFrame, expande
    listas de pontuações de cada fold em linhas separadas e tenta converter os valores para
    numéricos.

    Args:
        resultados (dict): Dicionário onde as chaves são nomes de modelos e os valores são
            dicionários com métricas de validação cruzada (ex.: 'fit_time', 'score_time',
            'test_r2', 'test_neg_mean_absolute_error', 'test_neg_root_mean_squared_error').
            Geralmente obtido de cross_validate ou GridSearchCV.cv_results_.

    Returns:
        pd.DataFrame: DataFrame com uma linha por fold e modelo, contendo colunas como 'model',
            'fit_time', 'score_time', 'time_seconds', 'test_r2', 'test_neg_mean_absolute_error',
            'test_neg_root_mean_squared_error', todas convertidas para tipo numérico quando
            possível.

    Examples:
        >>> resultados = {
        ...     'LinearRegression': {
        ...         'fit_time': [0.1, 0.2], 'score_time': [0.05, 0.06],
        ...         'test_r2': [0.8, 0.85]
        ...     },
        ...     'RandomForest': {
        ...         'fit_time': [0.5, 0.6], 'score_time': [0.1, 0.12],
        ...         'test_r2': [0.9, 0.88]
        ...     }
        ... }
        >>> df = organiza_resultados(resultados)
        >>> print(df[['model', 'time_seconds', 'test_r2']])
                  model  time_seconds  test_r2
        0  LinearRegression          0.15     0.80
        1  LinearRegression          0.26     0.85
        2      RandomForest          0.60     0.90
        3      RandomForest          0.72     0.88
    """