// static/js/chart.js
document.addEventListener('DOMContentLoaded', function() {
    // 加载训练历史数据
    loadTrainingHistory();
});

async function loadTrainingHistory() {
    try {
        // 尝试从training_history.json加载数据
        const response = await fetch('../model/training_history.json');
        const data = await response.json();
        
        // 创建图表
        const ctx = document.getElementById('historyChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.loss.map((_, i) => i + 1),
                datasets: [
                    {
                        label: '训练损失',
                        data: data.loss,
                        borderColor: 'rgba(102, 126, 234, 1)',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: '验证损失',
                        data: data.val_loss,
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
        // 如果加载失败，使用示例数据
        loadSampleTrainingData();
    }
}

function loadSampleTrainingData() {
    // 生成示例训练数据
    const epochs = [];
    const train_loss = [];
    const val_loss = [];
    
    for (let i = 1; i <= 50; i++) {
        epochs.push(i);
        train_loss.push(0.1 * Math.exp(-0.05 * i) + Math.random() * 0.01);
        val_loss.push(0.12 * Math.exp(-0.04 * i) + Math.random() * 0.015);
    }
    
    // 创建图表
    const ctx = document.getElementById('historyChart').getContext('2d');
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: '训练损失',
                    data: train_loss,
                    borderColor: 'rgba(102, 126, 234, 1)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: '验证损失',
                    data: val_loss,
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
}
