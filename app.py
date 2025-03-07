import os

from flask import Flask, render_template, request, send_file,jsonify
from reportlab.pdfgen import canvas
from werkzeug.utils import secure_filename
from datetime import datetime
import time  # ← 添加这行


app = Flask(__name__, static_folder='static', static_url_path='/static')  # ← 添加静态目录配置
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')  # ← 修改上传路径

# 保留模型接口位置
def predict_video(path):
    # 这里后续连接预测模型
    return {
        'images': ['demo1.jpg', 'demo2.jpg'],  # 示例数据
        'types': ['Type A', 'Type B'],
        'timestamps': ['00:01:23', '00:02:45']
    }

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': '未选择文件'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': '空文件名'}), 400

    # 增强文件类型验证
    if file and '.' in file.filename:
        ext = file.filename.rsplit('.', 1)[1].lower()
        if ext not in app.config['ALLOWED_EXTENSIONS']:
            return jsonify({'error': f'不支持 {ext} 格式'}), 400
        if file.mimetype not in ['video/mp4', 'video/avi', 'video/quicktime']:
            return jsonify({'error': '文件类型不匹配'}), 400

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(save_path)
            return jsonify({
                'video_url': f'/static/uploads/{filename}',
                'preview': f'/static/uploads/{filename}?t={int(time.time())}'
            })
        except Exception as e:
            return jsonify({'error': '存储失败'}), 500

    return jsonify({'error': '无效文件'}), 400

@app.route('/generate-report')
def generate_report():
    # 示例PDF生成
    pdf_path = "report.pdf"
    c = canvas.Canvas(pdf_path)
    c.drawString(100, 800, "ESD预测报告")
    # 这里添加实际数据
    c.save()
    return send_file(pdf_path, as_attachment=True)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # 这里调用预测模型
    return {
        'images': ['demo1.jpg', 'demo2.jpg'],
        'types': ['Type A', 'Type B'],
        'timestamps': ['00:01:23', '00:02:45']
    }

if __name__ == '__main__':
    # 确保所有必需目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('static/images', exist_ok=True)  # ← 新增预测结果图片目录
    app.run(debug=True)