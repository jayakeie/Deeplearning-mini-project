from flask import Flask, render_template, request
from deepface import DeepFace
import numpy as np
import cv2
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 웹캠 이미지인 경우
        if 'image' in request.form and request.form['image']:
            image_data = request.form['image'].split(',')[1]
            image_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            display_image = request.form['image']
        
        # 파일 업로드인 경우
        elif 'file' in request.files:
            file = request.files['file']
            np_arr = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            _, buffer = cv2.imencode('.jpg', img)
            display_image = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode()
        
        else:
            return render_template('result.html', error='이미지가 없습니다')
        
        # DeepFace 분석
        result = DeepFace.analyze(img_path=img,
                                 actions=['emotion', 'age', 'gender'],
                                 detector_backend='opencv',
                                 enforce_detection=False)
        
        if isinstance(result, list):
            result = result[0]
        
        # 결과 준비
        emotion_scores = {k: float(v) for k, v in result['emotion'].items()}
        
        return render_template('result.html',
                             emotion=result['dominant_emotion'],
                             age=int(result['age']),
                             gender=result['dominant_gender'],
                             scores=emotion_scores,
                             image=display_image)
        
    except Exception as e:
        return render_template('result.html',
                             error=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)