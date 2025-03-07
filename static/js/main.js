// 上传视频处理
document.getElementById('video-upload').addEventListener('change', async function(e) {
    const file = e.target.files[0];
    if (!file) return;

    // 添加上传监控逻辑
    let uploadStartTime = Date.now();
    let loaded = 0;

    const progressBar = document.querySelector('.progress-bar');
    const speedElement = document.querySelector('.speed');
    const timeElement = document.querySelector('.time');
    const percentageElement = document.querySelector('.percentage');

    // 在原有fetch请求中添加监控
    const response = await fetch('/upload', {
        method: 'POST',
        body: createFormDataWithProgress(file, (progress) => {
            const { loaded, total } = progress;
            const percentage = ((loaded / total) * 100).toFixed(1);
            const elapsed = (Date.now() - uploadStartTime) / 1000;
            const speed = (loaded / 1024 / 1024 / elapsed).toFixed(1); // MB/s
            const remaining = (total - loaded) / (loaded / elapsed);

            progressBar.style.width = `${percentage}%`;
            speedElement.textContent = `${speed} MB/s`;
            timeElement.textContent = `剩余时间: ${formatTime(remaining)}`;
            percentageElement.textContent = `${percentage}%`;
        })
    });
});

/*****************
 * 新增工具函数 *
 *****************/
function createFormDataWithProgress(file, onProgress) {
    const formData = new FormData();
    const xhr = new XMLHttpRequest();

    formData.append('video', file);

    xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
            onProgress({
                loaded: e.loaded,
                total: e.total
            });
        }
    });

    return xhr.send(formData);
}

function formatTime(seconds) {
    if (isNaN(seconds) || !isFinite(seconds)) return '--';
    const minutes = Math.floor(seconds / 60);
    seconds = Math.floor(seconds % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}
// 预测功能
async function startPrediction() {
    const videoPath = document.querySelector('video source')?.src
    if (!videoPath) return alert('请先上传视频')

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_path: videoPath })
        })
        const results = await response.json()

        // 渲染结果
        const grid = document.getElementById('results-grid')
        grid.innerHTML = results.images.map((img, i) => `
            <div class="result-card">
                <img src="/static/images/${img}" alt="预测结果">
                <div class="meta">
                    <span class="timestamp">${results.timestamps[i]}</span>
                    <span class="type-tag">${results.types[i]}</span>
                </div>
            </div>
        `).join('')

    } catch (error) {
        alert('预测失败: ' + error.message)
    }
}

// 生成报告
async function generateReport() {
    const btn = document.querySelector('.ins-btn.warning');
    const originalHTML = btn.innerHTML;

    try {
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 生成中...';
        btn.disabled = true;
        const response = await fetch('/generate-report')
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = 'analysis-report.pdf'
        a.click()
    } catch (error) {
        alert('生成报告失败: ' + error.message)
    }
    finally {
        btn.innerHTML = originalHTML;
        btn.disabled = false;
    }
}

// 重置功能
function resetAll() {
    document.getElementById('video-upload').value = ''
    document.getElementById('video-player').innerHTML = ''
    document.getElementById('results-grid').innerHTML = ''
}

