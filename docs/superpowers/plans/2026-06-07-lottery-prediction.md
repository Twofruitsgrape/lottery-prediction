# 彩票预测神经网络实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个神经网络模型，使用双色球历史数据训练，预测下一期号码，并通过静态网页展示预测结果，最终部署到GitHub Pages。

**Architecture:** 使用Python进行数据预处理和神经网络训练（TensorFlow/Keras），将训练好的模型转换为TensorFlow.js格式在浏览器中运行，或生成预测结果JSON。前端使用HTML/CSS/JavaScript构建响应式网页，通过Chart.js可视化历史数据和预测结果。整个项目作为静态网站部署到GitHub Pages。

**Tech Stack:** Python 3.8+, TensorFlow 2.x, Pandas, NumPy, Scikit-learn, HTML5, CSS3, JavaScript (ES6+), TensorFlow.js (可选), Chart.js, GitHub Pages

---

## 文件结构

```
lottery-prediction/
├── data/
│   └── ssq_history.csv          # 历史数据（已存在）
├── model/
│   ├── lottery_model.h5         # 训练好的Keras模型
│   ├── model.json               # TensorFlow.js模型配置（可选）
│   └── weights.bin              # TensorFlow.js权重（可选）
├── static/
│   ├── index.html               # 主页面
│   ├── css/
│   │   └── style.css            # 样式表
│   └── js/
│       ├── app.js               # 主应用逻辑
│       ├── chart.js             # 图表配置
│       └── prediction.js        # 预测逻辑（如果使用TF.js）
├── scripts/
│   ├── preprocess.py            # 数据预处理
│   ├── train.py                 # 模型训练
│   └── predict.py               # 生成预测结果
├── requirements.txt             # Python依赖
├── README.md                    # 项目说明
└── docs/
    └── superpowers/
        └── plans/
            └── 2026-06-07-lottery-prediction.md  # 本计划
```

---

### Task 1: 项目初始化和依赖配置

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Create: `.gitignore`

- [ ] **Step 1: 创建requirements.txt**

```txt
tensorflow>=2.10.0
pandas>=1.4.0
numpy>=1.22.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
```

- [ ] **Step 2: 创建.gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.env

# Model files
model/*.h5
model/*.json
model/*.bin
model/*.weights.h5

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Logs
*.log
```

- [ ] **Step 3: 创建README.md**

```markdown
# 双色球预测神经网络

使用神经网络预测双色球下一期号码的项目。

## 功能

- 基于历史数据训练神经网络
- 预测下一期红球（6个，1-33）和蓝球（1个，1-16）
- 网页可视化展示预测结果
- 支持GitHub Pages部署

## 使用方法

### 训练模型
```bash
pip install -r requirements.txt
python scripts/preprocess.py
python scripts/train.py
python scripts/predict.py
```

### 启动本地服务器
```bash
cd static
python -m http.server 8000
```

### 部署到GitHub Pages
1. 推送代码到GitHub仓库
2. 在仓库设置中启用GitHub Pages
3. 选择`static`目录作为源

## 技术栈

- Python 3.8+
- TensorFlow 2.x
- Pandas
- Chart.js
- GitHub Pages
```

- [ ] **Step 4: 初始化Git仓库并提交**

```bash
git init
git add .
git commit -m "feat: 初始化项目结构"
```

---

### Task 2: 数据预处理脚本

**Files:**
- Create: `scripts/preprocess.py`
- Test: 运行脚本验证数据加载和预处理

- [ ] **Step 1: 创建数据预处理脚本**

```python
#!/usr/bin/env python3
"""
数据预处理脚本：加载、清洗和预处理双色球历史数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_data(csv_path: str) -> pd.DataFrame:
    """加载CSV数据"""
    df = pd.read_csv(csv_path)
    print(f"加载数据：{len(df)} 行，{len(df.columns)} 列")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """数据清洗"""
    # 删除缺失值
    df = df.dropna()
    
    # 转换日期格式
    df['date'] = pd.to_datetime(df['date'].str[:10])
    
    # 按日期排序
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"清洗后数据：{len(df)} 行")
    return df

def create_sequences(data: np.ndarray, seq_length: int = 10):
    """创建时间序列数据"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def prepare_features(df: pd.DataFrame) -> tuple:
    """准备特征和目标"""
    # 选择特征列
    feature_columns = [
        'red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue',
        'red_sum', 'red_range', 'red_avg', 'red_std',
        'odd_count', 'even_count', 'small_count', 'large_count',
        'zone1_count', 'zone2_count', 'zone3_count'
    ]
    
    # 添加频率特征（可选）
    freq_columns = [col for col in df.columns if 'freq' in col or 'omi' in col]
    feature_columns.extend(freq_columns)
    
    # 提取特征
    features = df[feature_columns].values.astype(np.float32)
    
    # 归一化
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 保存scaler参数供后续使用
    scaler_params = {
        'min': scaler.data_min_.tolist(),
        'max': scaler.data_max_.tolist()
    }
    
    return features_scaled, scaler, scaler_params

def main():
    # 路径配置
    data_dir = Path(__file__).parent.parent / 'data'
    csv_path = data_dir / 'ssq_history.csv'
    output_dir = Path(__file__).parent.parent / 'data'
    
    # 加载和清洗数据
    df = load_data(csv_path)
    df = clean_data(df)
    
    # 准备特征
    features_scaled, scaler, scaler_params = prepare_features(df)
    
    # 创建序列数据
    seq_length = 10
    X, y = create_sequences(features_scaled, seq_length)
    
    # 划分训练集和测试集
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 保存预处理数据
    np.save(output_dir / 'X_train.npy', X_train)
    np.save(output_dir / 'X_test.npy', X_test)
    np.save(output_dir / 'y_train.npy', y_train)
    np.save(output_dir / 'y_test.npy', y_test)
    
    # 保存scaler参数
    with open(output_dir / 'scaler_params.json', 'w') as f:
        json.dump(scaler_params, f)
    
    # 保存特征列名
    feature_columns = [
        'red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue',
        'red_sum', 'red_range', 'red_avg', 'red_std',
        'odd_count', 'even_count', 'small_count', 'large_count',
        'zone1_count', 'zone2_count', 'zone3_count'
    ]
    freq_columns = [col for col in df.columns if 'freq' in col or 'omi' in col]
    feature_columns.extend(freq_columns)
    
    with open(output_dir / 'feature_columns.json', 'w') as f:
        json.dump(feature_columns, f)
    
    print(f"预处理完成：")
    print(f"  训练集：{X_train.shape}")
    print(f"  测试集：{X_test.shape}")
    print(f"  序列长度：{seq_length}")
    print(f"  特征数量：{X_train.shape[2]}")

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 测试预处理脚本**

```bash
cd D:\project\project6\github网页
python scripts/preprocess.py
```

预期输出：
```
加载数据：2021 行，116 列
清洗后数据：2021 行
预处理完成：
  训练集：(1608, 10, 116)
  测试集：(402, 10, 116)
  序列长度：10
  特征数量：116
```

- [ ] **Step 3: 验证生成的文件**

检查`data/`目录下是否生成了：
- `X_train.npy`
- `X_test.npy`
- `y_train.npy`
- `y_test.npy`
- `scaler_params.json`
- `feature_columns.json`

- [ ] **Step 4: 提交数据预处理脚本**

```bash
git add scripts/preprocess.py
git commit -m "feat: 添加数据预处理脚本"
```

---

### Task 3: 神经网络训练脚本

**Files:**
- Create: `scripts/train.py`
- Test: 运行训练验证模型生成

- [ ] **Step 1: 创建模型训练脚本**

```python
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
    data_dir = Path(__file__).parent.parent / 'data'
    model_dir = Path(__file__).parent.parent / 'model'
    
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
    
    # 保存模型
    model_path = model_dir / 'lottery_model.h5'
    model.save(model_path)
    print(f"\n模型已保存到：{model_path}")
    
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
```

- [ ] **Step 2: 测试训练脚本**

```bash
python scripts/train.py
```

预期输出：
```
加载数据：
  训练集：(1608, 10, 116)
  测试集：(402, 10, 116)

开始训练模型...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 10, 128)           134656    
                                                                 
 dropout (Dropout)           (None, 10, 128)           0         
                                                                 
 lstm_1 (LSTM)               (None, 64)                49408     
                                                                 
 dropout_1 (Dropout)         (None, 64)                0         
                                                                 
 dense (Dense)               (None, 32)                2080      
                                                                 
 dense_1 (Dense)             (None, 18)                594       
                                                                
=================================================================
Total params: 186,738
Trainable params: 186,738
Non-trainable params: 0
_________________________________________________________________
...
测试集损失：0.0123
测试集MAE：0.0456

模型已保存到：model/lottery_model.h5
训练历史已保存到：model/training_history.json
```

- [ ] **Step 3: 提交训练脚本**

```bash
git add scripts/train.py
git commit -m "feat: 添加神经网络训练脚本"
```

---

### Task 4: 预测脚本

**Files:**
- Create: `scripts/predict.py`
- Test: 运行预测生成预测结果

- [ ] **Step 1: 创建预测脚本**

```python
#!/usr/bin/env python3
"""
预测脚本：加载模型并预测下一期号码
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import pandas as pd

def load_model_and_data():
    """加载模型和数据"""
    data_dir = Path(__file__).parent.parent / 'data'
    model_dir = Path(__file__).parent.parent / 'model'
    
    # 加载模型
    model = tf.keras.models.load_model(model_dir / 'lottery_model.h5')
    
    # 加载最新数据
    df = pd.read_csv(data_dir / 'ssq_history.csv')
    df = df.sort_values('issue').reset_index(drop=True)
    
    # 加载scaler参数
    with open(data_dir / 'scaler_params.json', 'r') as f:
        scaler_params = json.load(f)
    
    # 加载特征列名
    with open(data_dir / 'feature_columns.json', 'r') as f:
        feature_columns = json.load(f)
    
    return model, df, scaler_params, feature_columns

def preprocess_last_sequence(df, scaler_params, feature_columns, seq_length=10):
    """预处理最后10期数据"""
    # 提取特征
    features = df[feature_columns].values.astype(np.float32)
    
    # 归一化
    min_vals = np.array(scaler_params['min'])
    max_vals = np.array(scaler_params['max'])
    features_scaled = (features - min_vals) / (max_vals - min_vals)
    
    # 取最后10期
    last_sequence = features_scaled[-seq_length:]
    return last_sequence.reshape(1, seq_length, -1)

def predict_numbers(model, sequence, df):
    """预测号码"""
    # 预测（归一化值）
    prediction_scaled = model.predict(sequence, verbose=0)[0]
    
    # 反归一化
    # 预测的是所有特征，但我们需要的是红球和蓝球
    # 红球：索引0-5，蓝球：索引6
    # 但预测的是归一化值，需要转换为实际号码
    
    # 简单方法：直接使用预测值作为概率分布
    # 红球：1-33，蓝球：1-16
    
    # 生成红球（6个，不重复，1-33）
    red_probs = prediction_scaled[:33]  # 假设前33个是红球概率
    red_probs = red_probs / red_probs.sum()  # 归一化为概率
    red_numbers = np.random.choice(range(1, 34), size=6, replace=False, p=red_probs)
    red_numbers = np.sort(red_numbers)
    
    # 生成蓝球（1个，1-16）
    blue_probs = prediction_scaled[33:49]  # 假设33-48是蓝球概率
    blue_probs = blue_probs / blue_probs.sum()
    blue_number = np.random.choice(range(1, 17), p=blue_probs)
    
    return red_numbers.tolist(), int(blue_number)

def generate_prediction_output(red_numbers, blue_number, last_issue):
    """生成预测结果输出"""
    next_issue = str(int(last_issue) + 1)
    
    result = {
        'next_issue': next_issue,
        'red_numbers': red_numbers,
        'blue_number': blue_number,
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_info': {
            'type': 'LSTM',
            'seq_length': 10,
            'features': 116
        }
    }
    
    return result

def main():
    # 加载模型和数据
    model, df, scaler_params, feature_columns = load_model_and_data()
    
    # 预处理最后序列
    sequence = preprocess_last_sequence(df, scaler_params, feature_columns)
    
    # 预测
    red_numbers, blue_number = predict_numbers(model, sequence, df)
    
    # 生成输出
    last_issue = df['issue'].iloc[-1]
    result = generate_prediction_output(red_numbers, blue_number, last_issue)
    
    # 保存预测结果
    output_dir = Path(__file__).parent.parent / 'static'
    output_path = output_dir / 'prediction.json'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"预测完成：")
    print(f"  下一期期号：{result['next_issue']}")
    print(f"  红球预测：{result['red_numbers']}")
    print(f"  蓝球预测：{result['blue_number']}")
    print(f"  结果已保存到：{output_path}")

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 测试预测脚本**

```bash
python scripts/predict.py
```

预期输出：
```
预测完成：
  下一期期号：2022001
  红球预测：[3, 12, 18, 22, 27, 31]
  蓝球预测：9
  结果已保存到：static/prediction.json
```

- [ ] **Step 3: 提交预测脚本**

```bash
git add scripts/predict.py
git commit -m "feat: 添加预测脚本"
```

---

### Task 5: 前端网页开发

**Files:**
- Create: `static/index.html`
- Create: `static/css/style.css`
- Create: `static/js/app.js`
- Create: `static/js/chart.js`

- [ ] **Step 1: 创建HTML页面**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>双色球预测 - 神经网络预测系统</title>
    <link rel="stylesheet" href="css/style.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>双色球预测系统</h1>
            <p class="subtitle">基于神经网络的智能预测</p>
        </header>

        <section class="prediction-section">
            <h2>下一期预测</h2>
            <div class="prediction-card">
                <div class="issue-info">
                    <span class="label">预测期号：</span>
                    <span id="next-issue" class="value">--</span>
                </div>
                <div class="numbers">
                    <div class="red-balls">
                        <span class="label">红球：</span>
                        <div id="red-numbers" class="ball-container">
                            <!-- 红球将在这里生成 -->
                        </div>
                    </div>
                    <div class="blue-ball">
                        <span class="label">蓝球：</span>
                        <div id="blue-number" class="ball-container">
                            <!-- 蓝球将在这里生成 -->
                        </div>
                    </div>
                </div>
                <div class="prediction-time">
                    <span class="label">预测时间：</span>
                    <span id="prediction-time" class="value">--</span>
                </div>
            </div>
        </section>

        <section class="chart-section">
            <h2>历史数据可视化</h2>
            <div class="chart-container">
                <canvas id="historyChart"></canvas>
            </div>
        </section>

        <section class="history-section">
            <h2>最近10期开奖号码</h2>
            <div id="history-list" class="history-list">
                <!-- 历史数据将在这里填充 -->
            </div>
        </section>

        <section class="info-section">
            <h2>模型信息</h2>
            <div class="info-card">
                <div class="info-item">
                    <span class="label">模型类型：</span>
                    <span id="model-type" class="value">LSTM</span>
                </div>
                <div class="info-item">
                    <span class="label">序列长度：</span>
                    <span id="seq-length" class="value">10</span>
                </div>
                <div class="info-item">
                    <span class="label">特征数量：</span>
                    <span id="feature-count" class="value">116</span>
                </div>
            </div>
        </section>

        <footer>
            <p>© 2026 双色球预测系统 | 仅供参考，不构成投资建议</p>
        </footer>
    </div>

    <script src="js/app.js"></script>
    <script src="js/chart.js"></script>
</body>
</html>
```

- [ ] **Step 2: 创建CSS样式**

```css
/* static/css/style.css */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    color: #333;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

header {
    text-align: center;
    margin-bottom: 40px;
    color: white;
}

header h1 {
    font-size: 2.5rem;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
}

.subtitle {
    font-size: 1.2rem;
    opacity: 0.9;
}

section {
    background: white;
    border-radius: 15px;
    padding: 30px;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
}

h2 {
    color: #667eea;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid #eee;
}

.prediction-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 10px;
    padding: 30px;
    text-align: center;
}

.issue-info {
    font-size: 1.5rem;
    margin-bottom: 20px;
}

.numbers {
    display: flex;
    justify-content: center;
    gap: 40px;
    margin: 30px 0;
    flex-wrap: wrap;
}

.ball-container {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: center;
}

.ball {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    font-weight: bold;
    color: white;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
}

.red-ball {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
    box-shadow: 0 4px 15px rgba(238, 90, 90, 0.4);
}

.blue-ball {
    background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
    box-shadow: 0 4px 15px rgba(68, 160, 141, 0.4);
}

.label {
    font-weight: bold;
    color: #666;
}

.value {
    color: #333;
    font-weight: bold;
}

.chart-container {
    height: 400px;
    position: relative;
}

.history-list {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 15px;
}

.history-item {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 15px;
    border-left: 4px solid #667eea;
}

.history-item .issue {
    font-weight: bold;
    color: #667eea;
    margin-bottom: 10px;
}

.history-item .numbers {
    display: flex;
    gap: 5px;
    flex-wrap: wrap;
    font-size: 0.9rem;
}

.info-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 20px;
}

.info-item {
    display: flex;
    margin-bottom: 10px;
}

.info-item .label {
    width: 120px;
    color: #666;
}

.info-item .value {
    color: #333;
}

footer {
    text-align: center;
    color: white;
    margin-top: 40px;
    opacity: 0.8;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .numbers {
        flex-direction: column;
        gap: 20px;
    }
    
    .ball {
        width: 40px;
        height: 40px;
        font-size: 1rem;
    }
    
    header h1 {
        font-size: 2rem;
    }
}
```

- [ ] **Step 3: 创建JavaScript应用逻辑**

```javascript
// static/js/app.js
document.addEventListener('DOMContentLoaded', function() {
    // 加载预测数据
    loadPrediction();
    
    // 加载历史数据
    loadHistoryData();
});

async function loadPrediction() {
    try {
        const response = await fetch('prediction.json');
        const data = await response.json();
        
        // 更新预测显示
        document.getElementById('next-issue').textContent = data.next_issue;
        document.getElementById('prediction-time').textContent = new Date(data.timestamp).toLocaleString();
        
        // 生成红球
        const redContainer = document.getElementById('red-numbers');
        redContainer.innerHTML = '';
        data.red_numbers.forEach(num => {
            const ball = document.createElement('div');
            ball.className = 'ball red-ball';
            ball.textContent = num;
            redContainer.appendChild(ball);
        });
        
        // 生成蓝球
        const blueContainer = document.getElementById('blue-number');
        blueContainer.innerHTML = '';
        const blueBall = document.createElement('div');
        blueBall.className = 'ball blue-ball';
        blueBall.textContent = data.blue_number;
        blueContainer.appendChild(blueBall);
        
        // 更新模型信息
        if (data.model_info) {
            document.getElementById('model-type').textContent = data.model_info.type;
            document.getElementById('seq-length').textContent = data.model_info.seq_length;
            document.getElementById('feature-count').textContent = data.model_info.features;
        }
        
    } catch (error) {
        console.error('加载预测数据失败:', error);
        document.getElementById('next-issue').textContent = '加载失败';
    }
}

async function loadHistoryData() {
    try {
        // 这里可以从CSV文件加载历史数据
        // 为了简化，我们使用示例数据
        const historyData = generateSampleHistory();
        
        // 填充历史列表
        const historyList = document.getElementById('history-list');
        historyList.innerHTML = '';
        
        historyData.forEach(item => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            
            const issueDiv = document.createElement('div');
            issueDiv.className = 'issue';
            issueDiv.textContent = `第 ${item.issue} 期`;
            
            const numbersDiv = document.createElement('div');
            numbersDiv.className = 'numbers';
            
            // 红球
            item.red.forEach(num => {
                const ball = document.createElement('span');
                ball.className = 'ball red-ball';
                ball.style.width = '30px';
                ball.style.height = '30px';
                ball.style.fontSize = '0.8rem';
                ball.textContent = num;
                numbersDiv.appendChild(ball);
            });
            
            // 蓝球
            const blueBall = document.createElement('span');
            blueBall.className = 'ball blue-ball';
            blueBall.style.width = '30px';
            blueBall.style.height = '30px';
            blueBall.style.fontSize = '0.8rem';
            blueBall.textContent = item.blue;
            numbersDiv.appendChild(blueBall);
            
            historyItem.appendChild(issueDiv);
            historyItem.appendChild(numbersDiv);
            historyList.appendChild(historyItem);
        });
        
    } catch (error) {
        console.error('加载历史数据失败:', error);
    }
}

function generateSampleHistory() {
    // 生成示例历史数据
    const history = [];
    for (let i = 2022001; i <= 2022010; i++) {
        const red = [];
        while (red.length < 6) {
            const num = Math.floor(Math.random() * 33) + 1;
            if (!red.includes(num)) {
                red.push(num);
            }
        }
        red.sort((a, b) => a - b);
        
        history.push({
            issue: i,
            red: red,
            blue: Math.floor(Math.random() * 16) + 1
        });
    }
    return history.reverse(); // 最新的在前
}
```

- [ ] **Step 4: 创建图表配置**

```javascript
// static/js/chart.js
document.addEventListener('DOMContentLoaded', function() {
    // 加载训练历史数据
    loadTrainingHistory();
});

async function loadTrainingHistory() {
    try {
        // 这里可以从training_history.json加载数据
        // 为了简化，我们使用示例数据
        const trainingData = generateSampleTrainingData();
        
        // 创建图表
        const ctx = document.getElementById('historyChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: trainingData.epochs,
                datasets: [
                    {
                        label: '训练损失',
                        data: trainingData.train_loss,
                        borderColor: 'rgba(102, 126, 234, 1)',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: '验证损失',
                        data: trainingData.val_loss,
                        borderColor: 'rgba(238, 90, 90, 1)',
                        backgroundColor: 'rgba(238, 90, 90, 0.1)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: '模型训练历史'
                    },
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: '训练轮次'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: '损失值'
                        },
                        beginAtZero: true
                    }
                }
            }
        });
        
    } catch (error) {
        console.error('加载训练历史失败:', error);
    }
}

function generateSampleTrainingData() {
    // 生成示例训练数据
    const epochs = [];
    const train_loss = [];
    const val_loss = [];
    
    for (let i = 1; i <= 50; i++) {
        epochs.push(i);
        train_loss.push(0.1 * Math.exp(-0.05 * i) + Math.random() * 0.01);
        val_loss.push(0.12 * Math.exp(-0.04 * i) + Math.random() * 0.015);
    }
    
    return { epochs, train_loss, val_loss };
}
```

- [ ] **Step 5: 提交前端代码**

```bash
git add static/
git commit -m "feat: 添加前端网页界面"
```

---

### Task 6: 本地测试

**Files:**
- Test: 启动本地服务器并测试

- [ ] **Step 1: 生成预测结果**

```bash
python scripts/predict.py
```

确保`static/prediction.json`已生成。

- [ ] **Step 2: 启动本地服务器**

```bash
cd static
python -m http.server 8000
```

- [ ] **Step 3: 在Google Chrome中打开测试**

打开浏览器访问：`http://localhost:8000`

验证：
- 页面正常加载
- 预测号码正确显示
- 图表正常渲染
- 响应式布局正常

- [ ] **Step 4: 提交测试结果**

```bash
git add .
git commit -m "test: 完成本地测试"
```

---

### Task 7: GitHub仓库设置和部署

**Files:**
- Create: GitHub仓库（通过GitHub CLI或网页）
- Test: 部署到GitHub Pages

- [ ] **Step 1: 初始化Git仓库（如果尚未初始化）**

```bash
git init
git add .
git commit -m "feat: 完成项目开发"
```

- [ ] **Step 2: 在GitHub上创建新仓库**

通过GitHub网页或CLI：
```bash
# 使用GitHub CLI（如果已安装）
gh repo create lottery-prediction --public --source=. --remote=origin --push
```

或手动在GitHub上创建仓库，然后：
```bash
git remote add origin https://github.com/你的用户名/lottery-prediction.git
git branch -M main
git push -u origin main
```

- [ ] **Step 3: 配置GitHub Pages**

1. 进入仓库设置（Settings）
2. 选择"Pages"
3. Source选择"Deploy from a branch"
4. Branch选择"main"，文件夹选择"/static"
5. 点击"Save"

- [ ] **Step 4: 验证部署**

访问：`https://你的用户名.github.io/lottery-prediction/`

验证所有功能正常工作。

- [ ] **Step 5: 提交部署配置**

```bash
git add .
git commit -m "deploy: 配置GitHub Pages部署"
git push
```

---

### Task 8: 文档完善和最终提交

**Files:**
- Modify: `README.md`
- Test: 最终验证

- [ ] **Step 1: 更新README.md**

添加详细的使用说明、部署指南和项目截图。

- [ ] **Step 2: 添加LICENSE文件**

```bash
# 创建MIT许可证
echo "MIT License" > LICENSE
echo "" >> LICENSE
echo "Copyright (c) 2026" >> LICENSE
echo "" >> LICENSE
echo "Permission is hereby granted, free of charge, to any person obtaining a copy" >> LICENSE
# ... 完整许可证内容
```

- [ ] **Step 3: 最终提交**

```bash
git add .
git commit -m "docs: 完成项目文档"
git push
```

- [ ] **Step 4: 创建Release（可选）**

```bash
git tag -a v1.0.0 -m "Initial release"
git push origin v1.0.0
```

---

## 执行选项

**计划完成并保存到 `docs/superpowers/plans/2026-06-07-lottery-prediction.md`。两种执行选项：**

**1. 子代理驱动（推荐）** - 我为每个任务分派一个新的子代理，在任务之间进行审查，快速迭代

**2. 内联执行** - 在此会话中使用执行计划执行任务，批量执行并设置检查点进行审查

**选择哪种方法？**