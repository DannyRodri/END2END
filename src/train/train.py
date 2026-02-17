import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline


MODEL_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
}

def obtener_modelos(params: dict):
    modelos = []

    for _, cfg in params["modeling"]["models"].items():
        model_class = MODEL_REGISTRY[cfg["class"]]
        model = model_class(**cfg.get("params", {}))

        grid = {
            "model": [model],
            **cfg.get("param_grid", {})
        }

        modelos.append(grid)

    return modelos



def entrenar_y_loggear(
    run_name,
    experiment_name,
    param_grid,
    preprocessor,
    X_train,
    y_train,
    cv=5
):

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", LogisticRegression())
        ])

        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("best_cv_accuracy", grid.best_score_)

        mlflow.sklearn.log_model(
            grid.best_estimator_,
            artifact_path="model"
        )

        return grid

