import numpy as np
from detectron2.structures import Instances

from utils import keypoint_names


class Instance:
    def __init__(self, *args):

        if len(args) > 1:
            raise TypeError("Instance() takes 1 positional argument but 2 were given")
        param = args[0]

        if isinstance(param, Instances):
            self.num_instances = len(param)
            self.instances = []
            boxes = np.asarray(param.get('pred_boxes').tensor, dtype=np.int32).tolist()
            scores = param.get('scores').tolist()
            keypoints = np.asarray(param.get('pred_keypoints'), dtype=np.int32).tolist()
            for n in range(self.num_instances):
                temp = dict()
                temp['area'] = (boxes[n][2] - boxes[n][0]) * (boxes[n][3] - boxes[n][1])
                temp['box'] = boxes[n]
                temp['score'] = scores[n]
                temp['keypoint'] = dict()
                # only the visible keypoints are stored
                for kpn, kp in zip(keypoint_names, keypoints[n]):
                    if kp[2] >= 0:
                        temp['keypoint'][kpn] = kp
                    else:
                        pass
                self.instances.append(temp)
            self.instances = sorted(self.instances, key=lambda i: i['area'], reverse=True)
        if isinstance(param, dict):
            self.num_instances = param['num_instances']
            self.instances = param['instances']

    def __len__(self) -> int:
        return self.num_instances

    def __dict__(self):
        return {"num_instances": self.num_instances, "instances": self.instances}

    def __json__(self):
        return {"num_instances": self.num_instances, "instances": self.instances}
