import tensorflow as tf
import mlflow


def build_model(params):
    activation=params['activation']
    filters=params['filters']
    number_of_layers=params['number_of_layers']
    dense_units=params['dense_units']
    
    inputs = tf.keras.Input(shape=(150, 1))
    x=(inputs)

    for layer in range(number_of_layers):
        x = tf.keras.layers.Conv1D(filters, 3, activation=activation)(x)
        x = tf.keras.layers.MaxPooling1D(3)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(dense_units, activation=activation)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def load_best_model():
    mlflow.set_experiment('Deployment')
    df=mlflow.search_runs()
    run_id = df.iloc[df['metrics.f1'].idxmax()]['run_id']
    model = mlflow.pyfunc.load_model("runs:/" + run_id + "/model")
    return model

