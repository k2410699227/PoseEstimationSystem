import cv2
import numpy as np

from instance import Instance


class Visualizer:
    def __init__(self):
        self.keypoint_connection_rules = [('left_ear', 'left_eye', (102, 204, 255)),
                                          ('right_ear', 'right_eye', (51, 153, 255)),
                                          ('left_eye', 'nose', (102, 0, 204)), ('nose', 'right_eye', (51, 102, 255)),
                                          ('left_shoulder', 'right_shoulder', (255, 128, 0)),
                                          ('left_shoulder', 'left_elbow', (153, 255, 204)),
                                          ('right_shoulder', 'right_elbow', (128, 229, 255)),
                                          ('left_elbow', 'left_wrist', (153, 255, 153)),
                                          ('right_elbow', 'right_wrist', (102, 255, 224)),
                                          ('left_hip', 'right_hip', (255, 102, 0)),
                                          ('left_hip', 'left_knee', (255, 255, 77)),
                                          ('right_hip', 'right_knee', (153, 255, 204)),
                                          ('left_knee', 'left_ankle', (191, 255, 128)),
                                          ('right_knee', 'right_ankle', (255, 195, 77))]

    def __call__(self, image, instances: Instance, show_box=True, show_prob=True):
        # print(instances.__dict__())
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
            for rule in self.keypoint_connection_rules:
                if self.exist_keypoints(keypoints, rule[0], rule[1]):
                    cv2.line(image, keypoints[rule[0]][:2], keypoints[rule[1]][:2], rule[2][::-1], 2 * scale_ratio)

            for i, p in enumerate(keypoints.values()):
                # print(p)
                if p[2] >= 0:
                    cv2.circle(image, (p[0], p[1]), 2 * scale_ratio, (0, 0, 255), -1)
                else:
                    pass

            if self.exist_keypoints(keypoints, "nose", "left_shoulder", "right_shoulder"):
                mid_shoulder = self.mid_point(keypoints["left_shoulder"], keypoints["right_shoulder"])
                cv2.line(image, keypoints["nose"][:2], mid_shoulder, (0, 0, 255), 2 * scale_ratio)

            if self.exist_keypoints(keypoints, "left_hip", "right_hip", "left_shoulder", "right_shoulder"):
                mid_shoulder = self.mid_point(keypoints["left_shoulder"], keypoints["right_shoulder"])
                mid_hip = self.mid_point(keypoints["left_hip"], keypoints["right_hip"])
                cv2.line(image, mid_hip, mid_shoulder, (0, 0, 255), 2 * scale_ratio)
        return image[:, :, ::-1]  # 将源BGR图像转为RGB

    @staticmethod
    def exist_keypoints(keypoints: dict, *args: str) -> bool:
        for kpn in args:
            if kpn in keypoints:
                continue
            else:
                return False
        return True

    @staticmethod
    def mid_point(start, end):
        x = (start[0] + end[0]) / 2
        y = (start[1] + end[1]) / 2
        return int(x), int(y)
