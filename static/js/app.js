// static/js/app.js - 主应用逻辑

// 全局变量
let historyData = [];
let currentPage = 1;
const itemsPerPage = 20;

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    initTabs();
    loadPrediction();
    loadHistoryData();
    initParamControls();
});

// 标签切换
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(btn.dataset.tab).classList.add('active');
        });
    });
}

// 加载预测数据
async function loadPrediction() {
    try {
        const response = await fetch('prediction.json');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        
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
        
        if (data.model_info) {
            document.getElementById('model-type').textContent = data.model_info.type;
        }
        
        updateStats(data);
    } catch (error) {
        console.error('加载预测数据失败:', error);
        // 显示错误提示
        document.getElementById('next-issue').textContent = '加载失败';
        document.getElementById('red-numbers').innerHTML = '<span style="color:#e74c3c">无法加载预测数据</span>';
        document.getElementById('blue-number').innerHTML = '';
    }
}

// 重新预测按钮 - 重新加载 prediction.json
document.getElementById('btn-refresh-prediction')?.addEventListener('click', () => {
    loadPrediction();
});
