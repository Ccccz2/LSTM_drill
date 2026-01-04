import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(data_path):
    """加载原始数据"""
    return pd.read_excel(data_path, engine='openpyxl')

def create_dataset(X, y, time_steps):
    """创建时间序列数据集"""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def preprocess_data(data, time_steps):
    """数据预处理流程（适配新输入输出）"""
    # 新输入特征（8个）
    features = data[[
        "单轴抗压强度(MPa)",
        "钻速(mm/s)",
        "转速(r/s)",
        "角速度(1/s)",
        "给进压力(MPa)",
        "推力(kN)",
        "扭矩(N·m)",
        "岩石可钻性指数"
    ]].values

    # 新输出目标（3个）
    target = data[[
        "比能的推力分量(N/mm2)",
        "比能的旋转分量(N/mm2)",
        "比能e(N/mm2)"
    ]].values

    # 检查数据量是否足够
    if len(features) <= time_steps:
        raise ValueError(f"数据量不足！需要至少 {time_steps + 1} 行数据，当前只有 {len(features)} 行")

    # 创建时间序列数据集（需确保 create_dataset 支持多输出）
    X, y = create_dataset(features, target, time_steps)

    # 数据标准化
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # 多目标标准化（每个输出单独标准化）
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)

    return X, y, scaler_X, scaler_y

def get_train_test_data(X, y, test_size=0.2):
    """划分训练集和测试集"""
    return train_test_split(X, y, test_size = test_size, shuffle = False)