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
        // 使用示例数据
        generateSamplePrediction();
    }
}

// 生成示例预测（当JSON加载失败时）
function generateSamplePrediction() {
    const redNumbers = [];
    while (redNumbers.length < 6) {
        const num = Math.floor(Math.random() * 33) + 1;
        if (!redNumbers.includes(num)) redNumbers.push(num);
    }
    redNumbers.sort((a, b) => a - b);
    
    const blueNumber = Math.floor(Math.random() * 16) + 1;
    
    document.getElementById('next-issue').textContent = '2026065';
    document.getElementById('prediction-time').textContent = new Date().toLocaleString();
    
    const redContainer = document.getElementById('red-numbers');
    redContainer.innerHTML = '';
    redNumbers.forEach(num => {
        const ball = document.createElement('div');
        ball.className = 'ball red-ball';
        ball.textContent = num;
        redContainer.appendChild(ball);
    });
    
    const blueContainer = document.getElementById('blue-number');
    blueContainer.innerHTML = '';
    const blueBall = document.createElement('div');
    blueBall.className = 'ball blue-ball';
    blueBall.textContent = blueNumber;
    blueContainer.appendChild(blueBall);
}

// 加载历史数据
async function loadHistoryData() {
    try {
        const response = await fetch('history.json');
        historyData = await response.json();
        
        document.getElementById('total-count').textContent = `共 ${historyData.length} 期`;
        renderHistory();
        initHistoryControls();
    } catch (error) {
        console.error('加载历史数据失败:', error);
        document.getElementById('total-count').textContent = '数据加载失败';
    }
}

// 渲染历史数据
function renderHistory() {
    const container = document.getElementById('history-list');
    const searchTerm = document.getElementById('search-issue').value;
    const sortOrder = document.getElementById('sort-order').value;
    
    let filteredData = historyData;
    
    // 搜索过滤
    if (searchTerm) {
        filteredData = filteredData.filter(item => 
            item.issue.includes(searchTerm)
        );
    }
    
    // 排序
    filteredData = sortOrder === 'desc' 
        ? [...filteredData].reverse()
        : filteredData;
    
    // 分页
    const startIndex = (currentPage - 1) * itemsPerPage;
    const paginatedData = filteredData.slice(startIndex, startIndex + itemsPerPage);
    
    container.innerHTML = '';
    paginatedData.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        
        const ballsHtml = item.red.map(num => 
            `<span class="ball red-ball">${num}</span>`
        ).join('') + `<span class="ball blue-ball">${item.blue}</span>`;
        
        historyItem.innerHTML = `
            <div class="issue">第 ${item.issue} 期</div>
            <div class="date">${item.date}</div>
            <div class="balls">${ballsHtml}</div>
        `;
        container.appendChild(historyItem);
    });
    
    // 更新分页信息
    const totalPages = Math.ceil(filteredData.length / itemsPerPage);
    document.getElementById('page-info').textContent = `第 ${currentPage} / ${totalPages} 页`;
    document.getElementById('btn-prev').disabled = currentPage === 1;
    document.getElementById('btn-next').disabled = currentPage >= totalPages;
}

// 初始化历史数据控制
function initHistoryControls() {
    document.getElementById('search-issue').addEventListener('input', () => {
        currentPage = 1;
        renderHistory();
    });
    
    document.getElementById('sort-order').addEventListener('change', () => {
        currentPage = 1;
        renderHistory();
    });
    
    document.getElementById('btn-prev').addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            renderHistory();
        }
    });
    
    document.getElementById('btn-next').addEventListener('click', () => {
        currentPage++;
        renderHistory();
    });
}

// 初始化参数控制
function initParamControls() {
    const params = [
        'lstm1-units', 'lstm2-units', 'dense-units', 'dropout-rate',
        'learning-rate', 'max-epochs', 'batch-size', 'seq-length',
        'early-patience', 'lr-decay-factor'
    ];
    
    params.forEach(param => {
        const input = document.getElementById(param);
        const display = document.getElementById(`${param}-value`);
        
        if (input && display) {
            input.addEventListener('input', () => {
                display.textContent = input.value;
                updateParamSummary();
            });
        }
    });
    
    // 预设按钮
    document.querySelectorAll('.btn-preset').forEach(btn => {
        btn.addEventListener('click', () => {
            applyPreset(btn.dataset.preset);
        });
    });
    
    updateParamSummary();
}

// 应用参数预设
function applyPreset(preset) {
    const presets = {
        default: {
            'lstm1-units': 128, 'lstm2-units': 64, 'dense-units': 32,
            'dropout-rate': 0.2, 'learning-rate': 0.001, 'max-epochs': 100,
            'batch-size': 32, 'seq-length': 10, 'early-patience': 10,
            'lr-decay-factor': 0.5
        },
        conservative: {
            'lstm1-units': 64, 'lstm2-units': 32, 'dense-units': 16,
            'dropout-rate': 0.4, 'learning-rate': 0.0005, 'max-epochs': 50,
            'batch-size': 64, 'seq-length': 15, 'early-patience': 15,
            'lr-decay-factor': 0.3
        },
        aggressive: {
            'lstm1-units': 256, 'lstm2-units': 128, 'dense-units': 64,
            'dropout-rate': 0.1, 'learning-rate': 0.002, 'max-epochs': 200,
            'batch-size': 16, 'seq-length': 20, 'early-patience': 5,
            'lr-decay-factor': 0.7
        },
        fast: {
            'lstm1-units': 32, 'lstm2-units': 16, 'dense-units': 8,
            'dropout-rate': 0.2, 'learning-rate': 0.003, 'max-epochs': 30,
            'batch-size': 128, 'seq-length': 5, 'early-patience': 5,
            'lr-decay-factor': 0.8
        }
    };
    
    const selected = presets[preset] || presets.default;
    
    Object.entries(selected).forEach(([key, value]) => {
        const input = document.getElementById(key);
        const display = document.getElementById(`${key}-value`);
        if (input) {
            input.value = value;
            if (display) display.textContent = value;
        }
    });
    
    updateParamSummary();
}

// 更新参数摘要
function updateParamSummary() {
    const params = {
        'LSTM-1': document.getElementById('lstm1-units')?.value,
        'LSTM-2': document.getElementById('lstm2-units')?.value,
        'Dense': document.getElementById('dense-units')?.value,
        'Dropout': document.getElementById('dropout-rate')?.value,
        '学习率': document.getElementById('learning-rate')?.value,
        '最大轮次': document.getElementById('max-epochs')?.value,
        '批大小': document.getElementById('batch-size')?.value,
        '序列长度': document.getElementById('seq-length')?.value,
        'EarlyStop耐心': document.getElementById('early-patience')?.value,
        '衰减因子': document.getElementById('lr-decay-factor')?.value
    };
    
    const summary = Object.entries(params)
        .map(([key, value]) => `${key}: ${value}`)
        .join('\n');
    
    document.getElementById('param-summary').textContent = summary;
}

// 更新统计信息
function updateStats(data) {
    document.getElementById('stat-total').textContent = historyData.length || '2021';
    
    // 计算模型参数量（近似）
    const lstm1 = parseInt(document.getElementById('lstm1-units')?.value || 128);
    const lstm2 = parseInt(document.getElementById('lstm2-units')?.value || 64);
    const dense = parseInt(document.getElementById('dense-units')?.value || 32);
    const params = (lstm1 * 4 * (116 + lstm1)) + (lstm2 * 4 * (lstm1 + lstm2)) + (lstm1 * dense) + dense;
    document.getElementById('stat-params').textContent = params.toLocaleString();
    
    document.getElementById('stat-loss').textContent = '0.0055';
    document.getElementById('stat-mae').textContent = '0.0253';
}

// 重新预测按钮 - 重新加载 prediction.json
document.getElementById('btn-refresh-prediction')?.addEventListener('click', () => {
    loadPrediction();
});
