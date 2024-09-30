import pytest
import tensorflow as tf
from time_series_classification.modeling.model import build_model

def test_build_model():
    params = {
        "activation": "relu",
        "filters": 32,
        "number_of_layers": 2,
        "dense_units": 16
    }

    model = build_model(params)

    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 140, 1)
    assert model.output_shape == (None, 5)
