#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from src import data_loading, model, training, utils
from src.config import CONFIG, DATA_PATH, RESULT_SAVE_DIR, BASE_DIR
from sklearn.preprocessing import StandardScaler

def main():
    # 初始化运行状态
    is_error = False
    error_msg = None

    try:
        # 1. 数据预处理
        raw_data = data_loading.load_data(DATA_PATH)
        X, y, scaler_X, scaler_y = data_loading.preprocess_data(raw_data, CONFIG['TIME_STEPS'])
        X_train, X_test, y_train, y_test = data_loading.get_train_test_data(X, y, CONFIG['TEST_SIZE'])
        scaler_y = StandardScaler()
        y_train_s = scaler_y.fit_transform(y_train)
        y_test_s = scaler_y.transform(y_test)

        # 2. 模型构建
        model_path = os.path.join(BASE_DIR, CONFIG['MODEL_NAME'])
        if os.path.exists(model_path):
            print(f"加载预训练模型: {model_path}")
            drill_model = model.load_saved_model(model_path)
        else:
            print("新建LSTM模型")
            drill_model = model.build_lstm_model((CONFIG['TIME_STEPS'], X.shape[2]))
        drill_model.summary()

        # 3. 模型训练
        history = training.train_model(
            drill_model,
            X_train, y_train_s,
            X_test, y_test_s,
            CONFIG)

        # 4. 模型评估
        mae, real_y_test, real_y_pred = training.evaluate_model(
            drill_model,
            X_test, y_test_s,
            scaler_y
        )
        print(f"测试集平均绝对误差(MAE): {mae:.2f} N/mm²")

        # 5. 保存结果
        save_dir = utils.get_save_dir(RESULT_SAVE_DIR)
        training.save_training_results(
            history = history,
            real_y_test = real_y_test,
            real_y_pred = real_y_pred,
            mae = mae,
            save_dir = save_dir,
            config = CONFIG,
            n_train_samples = len(X_train),
            n_test_samples = len(X_test),
        )

        # 6. 保存模型
        drill_model.save(os.path.join(save_dir, "drill_model.keras"))

    except Exception as e:
        is_error = True
        error_msg = str(e)
        error_dir = utils.get_save_dir(RESULT_SAVE_DIR, is_error=True)
        error_log = utils.save_error_log(error_dir, e)
        print(f"运行出错！错误日志已保存至: {error_log}")
        raise


if __name__ == "__main__":
    main()