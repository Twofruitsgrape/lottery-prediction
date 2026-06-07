// static/js/training.js - 神经网络训练模拟

// 训练状态
let isTraining = false;
let trainingTimer = null;
let currentEpoch = 0;

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    initTrainingControls();
});

function initTrainingControls() {
    document.getElementById('btn-start-training')?.addEventListener('click', startTraining);
    document.getElementById('btn-stop-training')?.addEventListener('click', stopTraining);
}

// 获取当前参数
function getCurrentParams() {
    return {
        lstm1Units: parseInt(document.getElementById('lstm1-units')?.value || 128),
        lstm2Units: parseInt(document.getElementById('lstm2-units')?.value || 64),
        denseUnits: parseInt(document.getElementById('dense-units')?.value || 32),
        dropoutRate: parseFloat(document.getElementById('dropout-rate')?.value || 0.2),
        learningRate: parseFloat(document.getElementById('learning-rate')?.value || 0.001),
        maxEpochs: parseInt(document.getElementById('max-epochs')?.value || 100),
        batchSize: parseInt(document.getElementById('batch-size')?.value || 32),
        seqLength: parseInt(document.getElementById('seq-length')?.value || 10),
        earlyPatience: parseInt(document.getElementById('early-patience')?.value || 10),
        lrDecayFactor: parseFloat(document.getElementById('lr-decay-factor')?.value || 0.5)
    };
}

// 开始训练
function startTraining() {
    if (isTraining) return;
    
    isTraining = true;
    currentEpoch = 0;
    
    const params = getCurrentParams();
    
    // 更新UI
    document.getElementById('btn-start-training').disabled = true;
    document.getElementById('btn-stop-training').disabled = false;
    updateTrainingStatus('训练中', 0, params.maxEpochs);
    clearLog();
    
    // 清空图表
    if (window.chartFunctions) {
        window.chartFunctions.clearTrainingChart();
    }
    
    addLog('info', '=== 开始训练 ===');
    addLog('info', `模型结构: LSTM(${params.lstm1Units}) -> LSTM(${params.lstm2Units}) -> Dense(${params.denseUnits})`);
    addLog('info', `学习率: ${params.learningRate}, 批大小: ${params.batchSize}`);
    addLog('info', `序列长度: ${params.seqLength}, Dropout: ${params.dropoutRate}`);
    addLog('info', '----------------------------');
    
    // 计算初始损失（基于参数）
    let currentLoss = calculateInitialLoss(params);
    let currentValLoss = currentLoss * 1.1;
    let bestLoss = Infinity;
    let patienceCounter = 0;
    let currentLR = params.learningRate;
    
    // 模拟训练循环
    trainingTimer = setInterval(() => {
        if (!isTraining || currentEpoch >= params.maxEpochs) {
            stopTraining();
            return;
        }
        
        currentEpoch++;
        
        // 模拟损失下降
        const decay = Math.exp(-0.05 * currentEpoch / (params.lstm1Units / 64));
        const noise = (Math.random() - 0.5) * 0.002;
        
        currentLoss = currentLoss * decay + noise;
        currentValLoss = currentLoss * (1.05 + Math.random() * 0.1) + noise * 1.2;
        
        // 确保损失不为负
        currentLoss = Math.max(0.001, currentLoss);
        currentValLoss = Math.max(0.001, currentValLoss);
        
        // Early stopping 检查
        if (currentValLoss < bestLoss) {
            bestLoss = currentValLoss;
            patienceCounter = 0;
        } else {
            patienceCounter++;
            if (patienceCounter >= params.earlyPatience) {
                addLog('warning', `Early stopping at epoch ${currentEpoch}`);
                stopTraining();
                return;
            }
        }
        
        // 学习率衰减
        if (currentEpoch % 10 === 0) {
            currentLR *= params.lrDecayFactor;
            addLog('info', `Epoch ${currentEpoch}: 学习率衰减至 ${currentLR.toFixed(6)}`);
        }
        
        // 更新图表
        if (window.chartFunctions) {
            window.chartFunctions.addTrainingPoint(currentEpoch, currentLoss, currentValLoss);
        }
        
        // 更新状态
        updateTrainingStatus('训练中', currentEpoch, params.maxEpochs);
        
        // 日志输出（每5轮输出一次）
        if (currentEpoch % 5 === 0) {
            addLog('success', `Epoch ${currentEpoch}/${params.maxEpochs} - loss: ${currentLoss.toFixed(4)} - val_loss: ${currentValLoss.toFixed(4)}`);
        }
        
        // 更新统计
        document.getElementById('stat-loss').textContent = currentLoss.toFixed(4);
        document.getElementById('stat-mae').textContent = (currentLoss * 5).toFixed(4);
        
    }, 100); // 每100ms模拟一个epoch
}

// 停止训练
function stopTraining() {
    isTraining = false;
    if (trainingTimer) {
        clearInterval(trainingTimer);
        trainingTimer = null;
    }
    
    // 更新UI
    document.getElementById('btn-start-training').disabled = false;
    document.getElementById('btn-stop-training').disabled = true;
    
    const params = getCurrentParams();
    updateTrainingStatus('已完成', currentEpoch, params.maxEpochs);
    
    addLog('info', '----------------------------');
    addLog('success', `训练完成! 共训练 ${currentEpoch} 轮`);
    addLog('info', '模型已保存（模拟）');
    
    // 生成新的预测
    generateNewPrediction(params);
}

// 计算初始损失（基于参数复杂度）
function calculateInitialLoss(params) {
    // 模型越复杂，初始损失可能越高（需要更多数据来训练）
    const complexity = (params.lstm1Units + params.lstm2Units + params.denseUnits) / 224;
    const baseLoss = 0.15 + complexity * 0.05;
    return baseLoss + Math.random() * 0.02;
}

// 生成新预测
function generateNewPrediction(params) {
    // 基于参数生成不同的预测结果
    const seed = params.lstm1Units * params.lstm2Units + params.learningRate * 10000;
    
    // 生成红球
    const redNumbers = [];
    const rng = mulberry32(seed);
    
    while (redNumbers.length < 6) {
        const num = Math.floor(rng() * 33) + 1;
        if (!redNumbers.includes(num)) {
            redNumbers.push(num);
        }
    }
    redNumbers.sort((a, b) => a - b);
    
    // 生成蓝球
    const blueNumber = Math.floor(rng() * 16) + 1;
    
    // 更新界面
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
    
    // 更新期号
    const nextIssue = document.getElementById('next-issue');
    const currentIssue = parseInt(nextIssue.textContent) || 2026064;
    nextIssue.textContent = currentIssue + 1;
    
    document.getElementById('prediction-time').textContent = new Date().toLocaleString();
    
    addLog('info', `新预测: 红球 ${redNumbers.join(', ')} | 蓝球 ${blueNumber}`);
}

// 简单的伪随机数生成器（确保相同参数产生相同结果）
function mulberry32(a) {
    return function() {
        a |= 0; a = a + 0x6D2B79F5 | 0;
        var t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
}

// 更新训练状态显示
function updateTrainingStatus(status, epoch, maxEpochs) {
    const statusEl = document.getElementById('training-status');
    if (statusEl) {
        statusEl.innerHTML = `
            <span>状态：${status}</span>
            <span>当前轮次：${epoch}/${maxEpochs}</span>
        `;
    }
}

// 日志功能
function addLog(type, message) {
    const logBox = document.getElementById('training-log');
    if (!logBox) return;
    
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logBox.appendChild(entry);
    logBox.scrollTop = logBox.scrollHeight;
}

function clearLog() {
    const logBox = document.getElementById('training-log');
    if (logBox) {
        logBox.innerHTML = '';
    }
}
