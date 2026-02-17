import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay



MODEL_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
}

def obtener_modelos(params: dict):
    modelos = []

    for _, cfg in params["models"].items():
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
    X_test,   
    y_test,     
    tags=None,
    cv=5
):
    
    # Asegurar limpieza de runs previos
    if mlflow.active_run():
        mlflow.end_run()

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        
        # 1. Registrar TAGS
        if tags:
            mlflow.set_tags(tags)

        # 2. Pipeline y Entrenamiento
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

        print(f"Entrenando modelo para: {run_name}...")
        grid.fit(X_train, y_train)

        # 3. Loggear Parámetros y el Mejor Score de Validación Cruzada
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("best_cv_accuracy", grid.best_score_)

        # 4. EVALUACIÓN (Predicciones en Test)
        # Usamos el mejor modelo encontrado para predecir en datos nuevos
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        # 5. Calcular Métricas
        # Nota: 'average' depende si es binario o multiclase. 
        # 'weighted' funciona bien para ambos casos generales.
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted') 
        test_recall = recall_score(y_test, y_pred, average='weighted')

        # 6. Loggear Métricas en MLflow
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("test_recall", test_recall)

        # 7. Generar y Loggear Matriz de Confusión (Como imagen)
        print("Generando matriz de confusión...")
        cm = confusion_matrix(y_test, y_pred)
        
        # Crear el gráfico
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=ax)
        plt.title(f"Confusion Matrix - {run_name}")
        
        # Guardar la imagen temporalmente y subirla a MLflow
        image_path = "confusion_matrix.png"
        plt.savefig(image_path)
        mlflow.log_artifact(image_path) # Se guarda en la pestaña 'Artifacts' de MLflow
        
        # Cerrar el plot para liberar memoria
        plt.close(fig)

        # 8. Guardar el modelo final
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            input_example=X_train.iloc[:5] # Opcional: ayuda a MLflow a entender el esquema
        )

        print(f"Run '{run_name}' finalizado con éxito.")
        return grid
