import io

import numpy as np
from PIL import Image

keypoint_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                  'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                  'right_knee', 'left_ankle', 'right_ankle']


def bytes_to_BGRImg(stream):
    with Image.open(io.BytesIO(stream)) as img:
        image = np.asarray(img)
        image = image[:, :, ::-1]
        return image


def exist_keypoints(keypoints: dict, *args: str) -> bool:
    for kpn in args:
        if kpn in keypoints:
            continue
        else:
            return False
    return True


def mid_point(start, end):
    x = (start[0] + end[0]) / 2
    y = (start[1] + end[1]) / 2
    return int(x), int(y)
