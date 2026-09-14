"""Adaptador de CatBoost GPU para FLAML.

CatBoost GPU no admite el callback de control de tiempo que FLAML
usa normalmente. Este adaptador ejecuta CatBoost directamente en GPU
y deja que FLAML evalue cada trial usando X_val / y_val.
"""

import time

import pandas as pd
from flaml import tune
from flaml.automl.model import CatBoostEstimator


CATBOOST_GPU_SEARCH_SPACE = {
    "catboost_gpu": {

        # Desactivar el early stopping heredado de FLAML.
        "early_stopping_rounds": {
            "domain": None,
        },

        "n_estimators": {
            "domain": tune.lograndint(300, 1501),
            "init_value": 600,
            "low_cost_init_value": 300,
        },

        "depth": {
            "domain": tune.randint(5, 10),
            "init_value": 6,
            "low_cost_init_value": 5,
        },

        "learning_rate": {
            "domain": tune.loguniform(0.02, 0.15),
            "init_value": 0.05,
            "low_cost_init_value": 0.10,
        },

        "l2_leaf_reg": {
            "domain": tune.loguniform(1.0, 20.0),
            "init_value": 3.0,
        },

        "random_strength": {
            "domain": tune.loguniform(1e-3, 5.0),
            "init_value": 1.0,
        },

        "border_count": {
            "domain": tune.choice([64, 128]),
            "init_value": 64,
        },
    }
}


class CatBoostGPUEstimator(CatBoostEstimator):
    """CatBoost multiclase usando la GPU 0."""

    def __init__(self, task="binary", **config):
        super().__init__(task, **config)

        # FLAML puede agregar este parametro desde el espacio
        # original de CatBoost. No queremos early stopping interno.
        self.params.pop("early_stopping_rounds", None)

        self.params.update({
            "task_type": "GPU",
            "devices": "0",
            "allow_writing_files": False,
            "random_seed": config.get("random_seed", 42),

            # Mostrar progreso cada 100 iteraciones para que
            # la terminal no parezca congelada.
            "verbose": 100,
        })

    def fit(
        self,
        X_train,
        y_train,
        budget=None,
        free_mem_ratio=0,
        **kwargs,
    ):
        del budget, free_mem_ratio

        kwargs.pop("is_retrain", None)
        kwargs.pop("groups", None)

        # No usar early stopping interno.
        kwargs.pop("early_stopping_rounds", None)
        kwargs.pop("eval_set", None)
        kwargs.pop("use_best_model", None)

        start_time = time.time()

        X_train = self._preprocess(X_train)

        if isinstance(X_train, pd.DataFrame):
            cat_features = list(
                X_train
                .select_dtypes(include="category")
                .columns
            )
        else:
            cat_features = []

        # Por seguridad, eliminar tambien del diccionario
        # de parametros del modelo.
        params = dict(self.params)
        params.pop("early_stopping_rounds", None)

        self._model = self.estimator_class(
            **params
        )

        self._model.fit(
            X_train,
            y_train,
            cat_features=cat_features,
            use_best_model=False,
            **kwargs,
        )

        return time.time() - start_time