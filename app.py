import os
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import time
import logging

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB

# 设置日志记录
logging.basicConfig(level=logging.DEBUG)

logging.debug(f"Static folder absolute path: {os.path.abspath(app.static_folder)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        logging.error("No video part in the request")
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['video']
    if file.filename == '':
        logging.error("Empty filename")
        return jsonify({'error': 'Empty filename'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            logging.debug(f"Saving file to {save_path}")
            file.save(save_path)
            logging.debug(f"File saved successfully to {save_path}")
            upload_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            return jsonify({
                'video_url': f'/static/uploads/{filename}',
                'preview': f'/static/uploads/{filename}?t={int(time.time())}',
                'filename': filename,
                'upload_time': upload_time
            })
        except Exception as e:
            logging.error(f"Error saving file: {e}")
            return jsonify({'error': f'Failed to save: {str(e)}'}), 500

    logging.error("Invalid file")
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/generate-report')
def generate_report(data):
    pdf_path = "report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # 获取上传视频的文件名
    filename = request.args.get('filename', 'Unknown')
    # 使用当前时间作为上传时间
    upload_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    # 标题
    title = Paragraph("<b><font size=18>ESD Prediction Report</font></b>", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))

    # 概述
    overview_text = ("<font size=12>This report presents the ESD prediction results for the uploaded video. "
                     "The following table summarizes the detected frames and their classifications.</font>")
    overview = Paragraph(overview_text, styles['Normal'])
    story.append(overview)
    story.append(Spacer(1, 12))

    # 上传信息
    upload_info = Paragraph(f"<font size=12><b>Filename:</b> {filename}<br/><b>Upload Time:</b> {upload_time}</font>", styles['Normal'])
    story.append(upload_info)
    story.append(Spacer(1, 12))


    # 表格样式
    table = Table(data, colWidths=[100, 100, 150])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),  # 表头背景色
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # 表头字体颜色
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # 居中对齐
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # 表头字体加粗
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),  # 表头行间距
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),  # 数据区背景色
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),  # 交替行颜色
        ('GRID', (0, 0), (-1, -1), 1, colors.black),  # 加边框
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # 结论
    conclusion_text = ("<font size=12><b>Conclusion:</b> The analysis of the detected frames suggests "
                       "patterns and trends in the ESD predictions. Further examination is recommended "
                       "to refine the classification model.</font>")
    conclusion = Paragraph(conclusion_text, styles['Normal'])
    story.append(conclusion)
    story.append(Spacer(1, 12))

    # 构建 PDF
    doc.build(story)

    return send_file(pdf_path, as_attachment=True)
@app.route('/predict', methods=['POST'])
def predict(data):
   
   
    return data

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    app.run(debug=True)