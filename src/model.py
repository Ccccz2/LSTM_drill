from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Input
from src import config

def build_lstm_model(input_shape):
    """构建LSTM模型"""
    model = Sequential([
        Input(shape = input_shape),
        LSTM(config.CONFIG['LSTM_layer']),
        Dense(config.CONFIG['Linear'], activation = config.CONFIG['Activation']),
        Dense(config.CONFIG['Output'])
    ])
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae'])
    return model

def load_saved_model(model_path):
    """加载预训练模型"""
    return load_model(model_path)
