from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error
from matplotlib import rcParams
import matplotlib.pyplot as plt
import os



def train_model(model, X_train, y_train, X_test, y_test, config):
    """训练模型"""
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(config['MODEL_NAME'], save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        epochs = config['EPOCHS'],
        batch_size = config['BATCH_SIZE'],
        validation_data = (X_test, y_test),
        callbacks = callbacks,
        verbose = 1
    )
    return history


def evaluate_model(model, X_test, y_test, scaler_y):
    """评估模型"""
    y_pred = model.predict(X_test)
    real_y_test = scaler_y.inverse_transform(y_test)
    real_y_pred = scaler_y.inverse_transform(y_pred)
    mae = mean_absolute_error(real_y_test, real_y_pred)
    return mae, real_y_test, real_y_pred


def save_training_results(history, real_y_test, real_y_pred, mae, save_dir, config, n_train_samples, n_test_samples):
    """保存训练结果和可视化"""
    # 确保目录存在
    os.makedirs(save_dir, exist_ok = True)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # 定义目标名称（与数据列顺序一致）
    target_names = ["比能的推力分量", "比能的旋转分量", "比能e (N/mm²)"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 为每条曲线分配颜色

    # 绘制真实值与预测值对比图
    plt.figure(figsize=(12, 8))
    for i in range(real_y_test.shape[1]):
        plt.plot(real_y_test[:, i],
                 label=f"{target_names[i]} (真实值)",
                 color=colors[i],
                 linewidth=2)
        plt.plot(real_y_pred[:, i],
                 label=f"{target_names[i]} (预测值)",
                 color=colors[i],
                 linestyle='--',
                 linewidth=1.5)

    plt.title("真实值与预测值对比")
    plt.xlabel("样本序号")
    plt.ylabel("数值")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在图表外侧
    plt.tight_layout()  # 避免图例遮挡
    plt.savefig(os.path.join(save_dir, "pred_vs_true.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # # 绘制真实值与预测值对比图（原始代码）
    # plt.figure(figsize=(10, 6))
    # plt.plot(real_y_test, label="真实值", linewidth=2)
    # plt.plot(real_y_pred, label="预测值", linestyle='--')
    # plt.savefig(os.path.join(save_dir, "pred_vs_true.png"))
    # plt.close()
    #
    #
    #
    # 保存训练曲线
    plt.plot(history.history['loss'], label = '训练损失')
    plt.plot(history.history['val_loss'], label = '验证损失')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "training_curve.png"))
    plt.close()
    #
    # # 保存预测对比图
    # plt.plot(real_y_test, label = "真实值")
    # plt.plot(real_y_pred, label = "预测值")
    # plt.legend()
    # plt.savefig(os.path.join(save_dir, "pred_vs_true.png"))
    # plt.close()


    # 保存训练日志
    with open(os.path.join(save_dir, "training_log.txt"), "w") as f:
        f.write(f"训练参数：\n")
        f.write(f"==模型结构==\n")
        f.write(f"LSTM层神经元个数：{config[ 'LSTM_layer']}\n")
        f.write(f"全连接层神经元个数：{config[ 'Linear']}\n")
        f.write(f"激活函数：{config['Activation']}\n")
        f.write(f"输出层神经元个数：{config[ 'Output']}\n\n")

        f.write(f"==超参数==\n")
        f.write(f"时间步长Time steps：{config['TIME_STEPS']}\n")
        f.write(f"训练轮次Epoch：{config['EPOCHS']}\n")
        f.write(f"批大小Batch size：{config['BATCH_SIZE']}\n")
        f.write(f"训练集占比Test size：{config['TEST_SIZE']}\n\n\n")


        f.write(f"训练结果：\n")
        f.write(f"测试集MAE：{mae:.2f}\n")
        f.write(f"最终验证损失：{min(history.history['val_loss']):.4f}\n\n\n")


        total_samples = n_train_samples + n_test_samples
        f.write(f"样本参数：{total_samples}\n")
        f.write(f"样本数量：{total_samples}\n")
        f.write(f"训练集数量：{n_train_samples} ({n_train_samples / total_samples:.1%})\n")
        f.write(f"测试集数量：{n_test_samples} ({n_test_samples / total_samples:.1%})\n\n")