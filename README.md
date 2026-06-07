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
