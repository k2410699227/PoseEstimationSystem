import io
import os
import time

import cv2
import torch
from PIL import Image
from detectron2.data.detection_utils import read_image
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit, send
from utils import *
from predictor import VisualizationDemo

app = Flask(__name__, template_folder='templates', static_url_path='/static', static_folder='templates')
socketio = SocketIO(app, cors_allowed_origins='*')
# socketio.init_app(app,)
# cap = cv2.VideoCapture(0)  # 打开摄像头（如果要处理视频文件，请提供文件路径）

paths = ['./test/' + i for i in os.listdir('test')][:5]
device = 'cuda'

model = torch.load("fcpose_res50.pth").to(device)
demo = VisualizationDemo(model)


@app.route('/')
def index():
    # global model
    # model = None
    return render_template('index.html')


def generate_frames():
    frame = cv2.imread("kun.jpg")
    for path in paths:
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)

        print("{}: detected {} instances in {:.2f}s".format(path, len(predictions["instances"]),
                                                            time.time() - start_time))
        # cv2.imwrite(path.replace(".jpg", "_res.jpg").replace("test", "test_result"),visualized_output.get_image()[:, :, ::-1])

        ret, buffer = cv2.imencode('.jpeg', visualized_output.get_image()[:, :, ::-1])
        frame = buffer.tobytes()
        # time.sleep(0.5)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


i = 0


@socketio.on('image')
def handle_image(data):
    global i
    i += 1
    # Process the image data here
    # print(type(data['data']))
    # image = Image.open(io.BytesIO(data['data']))
    image = bytes_to_BGRImg(data['data'])

    start_time = time.time()
    predictions, visualized_output = demo.run_on_image(image)

    print("detected {} instances in {:.2f}s".format(len(predictions["instances"]),
                                                    time.time() - start_time))
    # cv2.imwrite(path.replace(".jpg", "_res.jpg").replace("test", "test_result"),visualized_output.get_image()[:, :, ::-1])

    _, buffer = cv2.imencode('.jpeg', visualized_output.get_image()[:, :, ::-1])
    frame = buffer.tobytes()

    # time.sleep(2)
    return {'pose': frame}
    # emit('pose', {'pose': "hello world!"})


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=666, debug=True, allow_unsafe_werkzeug=True)
