# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from adet.modeling import OneStageDetector


class Predictor:
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
