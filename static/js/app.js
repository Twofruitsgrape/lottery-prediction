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
