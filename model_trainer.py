import os
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_trainer")

class ModelTrainer:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = None

    def build_model(self, input_shape):
        model = models.Sequential(name="LSTM_Model")
        model.add(layers.Input(shape=input_shape))
        model.add(layers.LSTM(64, return_sequences=True, name="lstm_1"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        model.add(layers.LSTM(32, return_sequences=False, name="lstm_2"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))

        model.add(layers.Dense(16, activation="relu", name="dense_1"))
        model.add(layers.Dense(1, name="output"))

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[
                "mae",
                "mean_absolute_percentage_error",
                tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error")
            ]
        )

        logger.info("Model architecture:\n%s", model.summary())
        return model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=32):
        self.model = self.build_model(input_shape=X_train.shape[1:])
        
        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )

        checkpoint_path = os.path.join(self.model_dir, "stock_price_model.h5")
        
        checkpoint = callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1
        )

        logger.info(f"Training artifacts saved to {checkpoint_path}")
        return history.history

    def load_model(self):
        model_path = os.path.join(self.model_dir, "stock_price_model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No saved model found at {model_path}")
        self.model = load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
        return self.model