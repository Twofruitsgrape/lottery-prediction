# GitHub 仓库创建和部署指南

## 步骤1：在GitHub上创建新仓库

1. 打开浏览器访问：https://github.com/new
2. 仓库名称填写：`lottery-prediction`
3. 描述填写：`双色球预测神经网络系统 - 基于LSTM的彩票号码预测`
4. 选择 **Public**（公开）
5. **不要**勾选任何初始化选项（README、.gitignore、license）
6. 点击 **Create repository**

## 步骤2：推送代码到GitHub

创建仓库后，在终端执行以下命令（替换 `你的用户名` 为你的GitHub用户名）：

```bash
# 添加远程仓库
git remote add origin https://github.com/你的用户名/lottery-prediction.git

# 推送代码
git branch -M main
git push -u origin main
```

## 步骤3：配置GitHub Pages

1. 进入你的仓库页面
2. 点击 **Settings**（设置）
3. 左侧菜单选择 **Pages**
4. Source 选择 **Deploy from a branch**
5. Branch 选择 **main**，文件夹选择 **/ static**
6. 点击 **Save**

## 步骤4：访问网站

部署完成后，访问：`https://你的用户名.github.io/lottery-prediction/`

## 注意事项

- 首次部署可能需要几分钟
- 确保 `static` 目录包含 `index.html`
- 预测数据 `prediction.json` 需要在仓库中
