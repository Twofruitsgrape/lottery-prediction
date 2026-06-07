// static/js/chart.js - 图表相关逻辑

// 训练历史图表
let trainingChart = null;

document.addEventListener('DOMContentLoaded', function() {
    initTrainingChart();
});

function initTrainingChart() {
    const ctx = document.getElementById('trainingChart');
    if (!ctx) return;
    
    trainingChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: '训练损失',
                    data: [],
                    borderColor: 'rgba(233, 69, 96, 1)',
                    backgroundColor: 'rgba(233, 69, 96, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: '验证损失',
                    data: [],
                    borderColor: 'rgba(243, 156, 18, 1)',
                    backgroundColor: 'rgba(243, 156, 18, 0.1)',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 300
            },
            plugins: {
                title: {
                    display: true,
                    text: '模型训练曲线',
                    color: '#fff'
                },
                legend: {
                    position: 'top',
                    labels: { color: '#fff' }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: '训练轮次 (Epoch)',
                        color: '#fff'
                    },
                    ticks: { color: '#aaa' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                y: {
                    title: {
                        display: true,
                        text: '损失值 (Loss)',
                        color: '#fff'
                    },
                    ticks: { color: '#aaa' },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    beginAtZero: true
                }
            }
        }
    });
    
    // 尝试加载已有训练历史
    loadTrainingHistory();
}

async function loadTrainingHistory() {
    try {
        const response = await fetch('training_history.json');
        const data = await response.json();
        
        if (data.loss && data.loss.length > 0) {
            updateTrainingChart(data.loss, data.val_loss);
        }
    } catch (error) {
        console.log('使用默认训练数据');
        // 使用示例数据
        const sampleLoss = generateSampleLoss(50, 0.1, 0.005);
        const sampleValLoss = generateSampleLoss(50, 0.12, 0.006);
        updateTrainingChart(sampleLoss, sampleValLoss);
    }
}

function generateSampleLoss(epochs, startLoss, endLoss) {
    const losses = [];
    for (let i = 0; i < epochs; i++) {
        const progress = i / (epochs - 1);
        const loss = startLoss * Math.exp(-3 * progress) + endLoss + (Math.random() - 0.5) * 0.005;
        losses.push(Math.max(endLoss, loss));
    }
    return losses;
}

function updateTrainingChart(trainLoss, valLoss) {
    if (!trainingChart) return;
    
    const labels = trainLoss.map((_, i) => i + 1);
    
    trainingChart.data.labels = labels;
    trainingChart.data.datasets[0].data = trainLoss;
    trainingChart.data.datasets[1].data = valLoss || trainLoss.map(v => v * 1.1);
    trainingChart.update();
}

// 添加新的训练数据点
function addTrainingPoint(epoch, trainLoss, valLoss) {
    if (!trainingChart) return;
    
    trainingChart.data.labels.push(epoch);
    trainingChart.data.datasets[0].data.push(trainLoss);
    trainingChart.data.datasets[1].data.push(valLoss);
    trainingChart.update('none'); // 无动画更新，提高性能
}

// 清空图表
function clearTrainingChart() {
    if (!trainingChart) return;
    
    trainingChart.data.labels = [];
    trainingChart.data.datasets[0].data = [];
    trainingChart.data.datasets[1].data = [];
    trainingChart.update();
}

// 导出函数供其他脚本使用
window.chartFunctions = {
    updateTrainingChart,
    addTrainingPoint,
    clearTrainingChart
};
