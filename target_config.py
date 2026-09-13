"""Objetivo comun: cinco clases de cambio porcentual del precio a 7 dias."""

import numpy as np
import pandas as pd

CLASSIFICATION_TARGET = "target_price_direction_7d"
SMALL_CHANGE_PCT = 3.0
LARGE_CHANGE_PCT = 10.0
# Los prefijos conservan el orden ordinal al ordenar las clases en los modelos.
PRICE_DIRECTION_CLASSES = (
    "0_baja_mucho", "1_baja_un_poco", "2_se_mantiene",
    "3_sube_un_poco", "4_sube_mucho",
)
PRICE_DIRECTION_DISPLAY = ("Baja++", "Baja", "Neutro", "Sube", "Sube++")
_DISPLAY_BY_CLASS = dict(zip(PRICE_DIRECTION_CLASSES, PRICE_DIRECTION_DISPLAY))


def price_direction_display_labels(classes):
    """Nombres cortos, en orden, para reportes y graficos."""
    return [_DISPLAY_BY_CLASS.get(str(value), str(value)) for value in classes]


def price_direction_labels(price, future_price):
    """Usar precios originales, antes de imputar o escalar las features."""
    if not 0 <= SMALL_CHANGE_PCT < LARGE_CHANGE_PCT:
        raise ValueError("Los umbrales deben cumplir 0 <= poco < mucho.")
    price = pd.to_numeric(price, errors="coerce")
    future_price = pd.to_numeric(future_price, errors="coerce")
    valid = np.isfinite(price) & np.isfinite(future_price) & (price > 0) & (future_price >= 0)
    change_pct = ((future_price - price) / price.where(valid) * 100).round(10)
    labels = np.select(
        [change_pct <= -LARGE_CHANGE_PCT,
         change_pct < -SMALL_CHANGE_PCT,
         change_pct >= LARGE_CHANGE_PCT,
         change_pct > SMALL_CHANGE_PCT],
        [PRICE_DIRECTION_CLASSES[0], PRICE_DIRECTION_CLASSES[1],
         PRICE_DIRECTION_CLASSES[4], PRICE_DIRECTION_CLASSES[3]],
        default=PRICE_DIRECTION_CLASSES[2],
    )
    return pd.Series(labels, index=price.index).where(valid)
