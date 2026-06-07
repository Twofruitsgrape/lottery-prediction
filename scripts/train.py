#!/usr/bin/env python3
"""
神经网络训练脚本：构建、训练和保存预测模型
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json

def build_model(input_shape: tuple, output_size: int) -> keras.Model:
    """构建LSTM模型"""
    model = keras.Sequential([
        keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(output_size, activation='sigmoid')  # 归一化输出
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """训练模型"""
    model = build_model(X_train.shape[1:], y_train.shape[1])
    
    # 打印模型结构
    model.summary()
    
    # 训练
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    return model, history

def main():
    # 路径配置
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    model_dir = base_dir / 'model'
    
    # 加载预处理数据
    X_train = np.load(data_dir / 'X_train.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    print(f"加载数据：")
    print(f"  训练集：{X_train.shape}")
    print(f"  测试集：{X_test.shape}")
    
    # 训练模型
    print("\n开始训练模型...")
    model, history = train_model(
        X_train, y_train,
        X_test, y_test,
        epochs=100,
        batch_size=32
    )
    
    # 评估模型
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n测试集损失：{test_loss:.4f}")
    print(f"测试集MAE：{test_mae:.4f}")
    
    # 保存模型（使用新的Keras格式）
    model_path = model_dir / 'lottery_model.keras'
    model.save(model_path)
    print(f"\n模型已保存到：{model_path}")
    
    # 同时保存权重（兼容旧版本）
    weights_path = model_dir / 'lottery_weights.weights.h5'
    model.save_weights(weights_path)
    print(f"权重已保存到：{weights_path}")
    
    # 保存训练历史
    history_path = model_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump({
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'mae': history.history['mae'],
            'val_mae': history.history['val_mae']
        }, f)
    
    print(f"训练历史已保存到：{history_path}")

if __name__ == '__main__':
    main()
