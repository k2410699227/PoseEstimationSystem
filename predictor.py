# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import matplotlib.pyplot as plt
from adet.modeling import OneStageDetector

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

from adet.utils.visualizer import TextVisualizer
from new_predictor import MyPredictor
from utils import *


class Predicter:
    def __init__(self, model, device=None):
        self.model = model
        if device is None:
            if torch.cuda.is_available():
                self.model.to('cuda')
            else:
                self.model.to('cpu')
        else:
            self.model.to(device)

    def __call__(self, image, original_shape, threshold=0.5):
        height, width = original_shape
        input = None
        if isinstance(self.model, OneStageDetector):
            if threshold != self.model.proposal_generator.fcos.fcos_outputs.pre_nms_thresh_test:
                self.model.proposal_generator.fcos.fcos_outputs.pre_nms_thresh_test = threshold
            input = {"image": image, "height": height, "width": width}
            predictions = self.model([input])[0]
            return predictions
        else:
            input = image
            predictions = self.model(input)
            return predictions
