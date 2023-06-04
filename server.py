import os
import time

import cv2
from flask import Flask, render_template
from flask_socketio import SocketIO

from pose_estimation import PoseEstimation
from utils import bytes_to_BGRImg


class Server:
    def __init__(self, model, device):
        self.app = Flask(__name__, template_folder='templates', static_url_path='/static', static_folder='templates')
        self.socketio = SocketIO(self.app, cors_allowed_origins='*', max_http_buffer_size=1024 * 1024 * 10)
        self.register_router()
        self.estimation = PoseEstimation(model, device=device)

    def register_router(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/image')
        def image():
            return render_template('image.html')

        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('Client disconnected')

        @self.socketio.on('image')
        def handle_image(data):
            # return {'pose': data['data']}

            # Process the image data here
            start = time.time()
            image = bytes_to_BGRImg(data['data'])
            print(image.shape, end='\t')
            # print("encodeInput_time: {:.2f}s".format(time.time() - start), end='\t')

            start_time = time.time()
            instances, visualized_output = self.estimation.run_on_image(image, short_edge=data['short_edge'],
                                                                        threshold=data['threshold'] / 100,
                                                                        show_box=not data["no_box"],
                                                                        show_prob=not data['no_prob'])

            print("detected {} instances in {:.4f}s"
                  "".format(len(instances),
                            time.time() - start_time), end='\t')
            start = time.time()
            encoding = '.jpg'
            _, buffer = cv2.imencode(encoding, visualized_output[:, :, ::-1])
            frame = buffer.tobytes()
            print("encodeOutput_time: {:.4f}s".format(time.time() - start), end='\n')
            # frame = data['data']

            return {'pose': frame, "instances": instances.__dict__(), "encoding": encoding}

        @self.socketio.on('ping')
        def ping(data):
            return {'data': os.urandom(1024 * 1024 * 5)}

    def run(self, **kwargs):
        print(kwargs)
        self.socketio.run(self.app, **kwargs)
