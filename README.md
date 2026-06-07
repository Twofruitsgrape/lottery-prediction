# 双色球预测神经网络

基于LSTM神经网络的双色球号码预测系统，通过历史数据训练模型，预测下一期开奖号码。

## 功能特点

- **神经网络预测**：使用LSTM（长短期记忆网络）分析历史数据
- **数据可视化**：图表展示模型训练过程和预测结果
- **响应式设计**：支持桌面和移动设备访问
- **GitHub Pages部署**：静态网页可直接在GitHub上运行

## 项目结构

```
lottery-prediction/
├── data/                    # 数据目录
│   ├── ssq_history.csv     # 历史开奖数据
│   ├── X_train.npy         # 训练集特征
│   ├── X_test.npy          # 测试集特征
│   ├── y_train.npy         # 训练集标签
│   ├── y_test.npy          # 测试集标签
│   ├── scaler_params.json  # 归一化参数
│   └── feature_columns.json # 特征列名
├── model/                   # 模型目录
│   ├── lottery_model.keras # 训练好的模型
│   ├── lottery_weights.weights.h5 # 模型权重
│   └── training_history.json # 训练历史
├── scripts/                 # 脚本目录
│   ├── preprocess.py       # 数据预处理
│   ├── train.py            # 模型训练
│   └── predict.py          # 预测生成
├── static/                  # 静态网页目录
│   ├── index.html          # 主页面
│   ├── prediction.json     # 预测结果
│   ├── css/style.css       # 样式表
│   └── js/                 # JavaScript文件
├── requirements.txt        # Python依赖
├── README.md               # 项目说明
├── DEPLOYMENT.md           # 部署指南
└── LICENSE                 # 许可证
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据预处理

```bash
python scripts/preprocess.py
```

### 3. 训练模型

```bash
python scripts/train.py
```

### 4. 生成预测

```bash
python scripts/predict.py
```

### 5. 启动本地服务器

```bash
cd static
python -m http.server 8000
```

访问 http://localhost:8000 查看结果

## 技术栈

- **后端**：Python 3.8+, TensorFlow 2.x, Pandas, NumPy, Scikit-learn
- **前端**：HTML5, CSS3, JavaScript, Chart.js
- **部署**：GitHub Pages

## 模型说明

- **模型类型**：LSTM（长短期记忆网络）
- **输入特征**：116个（包含红球、蓝球、频率、遗漏值等）
- **序列长度**：10期
- **输出**：预测下一期的红球（6个，1-33）和蓝球（1个，1-16）

## 部署到GitHub Pages

详见 [DEPLOYMENT.md](DEPLOYMENT.md)

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。彩票开奖结果是随机的，请理性购彩。

## 许可证

MIT License - 详见 [LICENSE](LICENSE)
