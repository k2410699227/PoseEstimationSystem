import cv2
import torch
import time
import torch.nn.functional as F
from adet.modeling import OneStageDetector

from transform import Transform
from predictor import Predictor
from visualizer import Visualizer
from instance import Instance
from utils import *


class PoseEstimation(object):
    def __init__(self, model, device=None, **kwargs):

        if device is not None:
            self.model = model.to(device)
        else:
            self.model = model
        self.model.eval()
        self.transform = Transform()
        self.visualizer = Visualizer()
        self.predicter = Predictor(model, device)

    def run_on_image(self, original_image, short_edge, threshold, show_box, show_prob):
        with torch.no_grad():
            instances = self.prediction_on_image(original_image, short_edge, threshold)

            visualized_output = self.visualizer(original_image, instances, show_box, show_prob)

        return instances, visualized_output

    def prediction_on_image(self, original_image, short_edge, threshold):
        image = self.transform(original_image, short_edge=short_edge)
        start = time.time()
        prediction = self.predicter(image, original_image.shape[:2], threshold)
        print("prediction_time: {:.4f}s".format(time.time() - start), end='\t')
        instances = Instance(prediction["instances"].to("cpu"))
        return instances


class PoseEstimation_One(object):
    def __init__(self, model, device=None, **kwargs):

        if device is not None:
            self.model = model.to(device)
        else:
            self.model = model
        self.model.eval()

    # 进行姿态估计的入口
    def run_on_image(self, image, short_edge=512, threshold=0.5):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
            :param threshold:
            :param visualize_output:
            :param image:
            :param short_edge:
        """

        if threshold != self.model.proposal_generator.fcos.fcos_outputs.pre_nms_thresh_test:
            if isinstance(self.model, OneStageDetector):
                self.model.proposal_generator.fcos.fcos_outputs.pre_nms_thresh_test = threshold
        start = time.time()
        # new_image = image.copy()
        if any(x < 0 for x in image.strides):
            image = np.ascontiguousarray(image)
        predictions = self.predict(image, short_edge)
        print("prediction_time: {:.4f}s".format(time.time() - start), end='\t')
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        # image = image[:, :, ::-1]
        start = time.time()

        instances = Instance(predictions["instances"].to("cpu"))

        return instances

    # 对图片进行预处理并调整图像的短边大小，保持宽高比不变
    @staticmethod
    def transform(img, short_edge):
        img = torch.from_numpy(img.astype("float32")).to("cuda")
        shape = list(img.shape)
        shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
        img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw

        h, w = shape[:2]
        size = short_edge * 1.0
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size

        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        img = F.interpolate(
            img, (newh, neww), mode="nearest"
        )
        shape[:2] = (newh, neww)
        # ret = img.permute(2, 3, 0, 1).view(shape)  # nchw -> hw(c)
        ret = img.squeeze(0)  #
        return ret

    # 对传入的图片进行姿态估计
    def predict(self, original_image, short_edge=512):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
                :param short_edge:
        """

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            start = time.time()

            height, width = original_image.shape[:2]

            # image = self.transform.get_transform(original_image).apply_image(original_image)
            image = self.transform(original_image, short_edge=short_edge)
            # image = torch.from_numpy(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            print("preinfer_time: {:.4f}s".format(time.time() - start), end='\t')
            start = time.time()
            predictions = self.model([inputs])[0]
            print("infer_time: {:.4f}s".format(time.time() - start), end='\t')
            return predictions

    # 可视化预测结果
    @staticmethod
    def visualize_output(image, instances: Instance, show_box=True, show_prob=True):
        if any(x < 0 for x in image.strides):
            image = np.ascontiguousarray(image)
        if len(instances) != instances.num_instances:
            raise AttributeError("实例预测发生错误")
        for ins in instances.instances[:]:
            # start = time.time()
            box = ins['box']
            keypoints: dict = ins['keypoint']
            scale_ratio = 1 if (round(np.amax(image.shape) / 1000) == 0) else round(np.amax(image.shape) / 1000)
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.5 * scale_ratio
            thickness = 1
            score = "{:}%".format(round(100 * ins['score']))
            text_size, _ = cv2.getTextSize(score, font, font_scale, thickness)
            if show_box:
                cv2.rectangle(image, box[:2], box[2:], (0, 200, 0), 1 * scale_ratio)
                if show_prob:
                    cv2.rectangle(image, box[:2], (box[0] + text_size[0], box[1] + text_size[1]), (0, 0, 0), -1)
                    cv2.putText(image, score, (box[0], box[1] + text_size[1]), font, font_scale, (0, 150, 180),
                                thickness)
            for rule in Visualizer.keypoint_connection_rules:
                if exist_keypoints(keypoints, rule[0], rule[1]):
                    cv2.line(image, keypoints[rule[0]][:2], keypoints[rule[1]][:2], rule[2][::-1], 2 * scale_ratio)

            for i, p in enumerate(keypoints.values()):
                # print(p)
                if p[2] >= 0:
                    cv2.circle(image, (p[0], p[1]), 2 * scale_ratio, (0, 0, 255), -1)
                else:
                    pass

            if exist_keypoints(keypoints, "nose", "left_shoulder", "right_shoulder"):
                mid_shoulder = mid_point(keypoints["left_shoulder"], keypoints["right_shoulder"])
                cv2.line(image, keypoints["nose"][:2], mid_shoulder, (0, 0, 255), 2 * scale_ratio)

            if exist_keypoints(keypoints, "left_hip", "right_hip", "left_shoulder", "right_shoulder"):
                mid_shoulder = mid_point(keypoints["left_shoulder"], keypoints["right_shoulder"])
                mid_hip = mid_point(keypoints["left_hip"], keypoints["right_hip"])
                cv2.line(image, mid_hip, mid_shoulder, (0, 0, 255), 2 * scale_ratio)
        return image[:, :, ::-1]  # 将源BGR图像转为RGB
