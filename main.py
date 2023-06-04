import argparse
import io
import json
import os
import time

import cv2
import torch
from PIL import Image

from adet.modeling.one_stage_detector import OneStageDetector
from detectron2.data.detection_utils import read_image
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit, send
from utils import *

from server import Server

# socketio.init_app(app,)
# cap = cv2.VideoCapture(0)  # 打开摄像头（如果要处理视频文件，请提供文件路径）

paths = ['./test/' + i for i in os.listdir('test')][:5]
device = 'cuda'
pretrained_model = "model/" + "fcpose_res50.pth"


def get_args():
    parser = argparse.ArgumentParser(description="Pose Estimation")

    parser.add_argument("--model", type=str, default="./model/fcpose_res50.pth",
                        help="Pretrained model for pose estimation")
    parser.add_argument("--device", type=str, help="Device for prediction, default: cuda,if available")
    parser.add_argument("--port", type=int, default=6006, help="The port for the server")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    model = torch.load(args.model)
    server = Server(model, args.device)
    server.run(host='0.0.0.0', port=args.port, debug=True, allow_unsafe_werkzeug=True)
