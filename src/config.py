import os
from datetime import datetime

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/raw/cz_drill.xlsx")
RESULT_SAVE_DIR = os.path.join(BASE_DIR, "results")

# 模型参数
CONFIG = {
    'LSTM_layer': 64,        # LSTM层神经元个数
    'Linear': 32,            # 全连接层神经元个数
    'Activation': 'relu',    # 激活函数
    'Output': 3,             # 输出层神经元个数


    'TIME_STEPS': 20,
    'EPOCHS': 50,
    'BATCH_SIZE': 16,
    'TEST_SIZE': 0.2,
    'MODEL_NAME': 'best_model.keras'
}