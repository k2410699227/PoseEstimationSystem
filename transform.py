import torch
import torch.nn.functional as F


class Transform:
    def __init__(self):
        pass

    def __call__(self, img, short_edge=512):
        img = torch.from_numpy(img.astype("float32")).to("cuda")
        shape = list(img.shape)
        shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
        img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw

        h, w = shape[:2]
        neww, newh = self.get_new_size(h, w, short_edge)

        img = F.interpolate(
            img, (newh, neww), mode="nearest"
        )
        shape[:2] = (newh, neww)
        # ret = img.permute(2, 3, 0, 1).view(shape)  # nchw -> hw(c)
        ret = img.squeeze(0)  #
        return ret

    @staticmethod
    def get_new_size(height, width, short_edge):
        size = short_edge * 1.0
        scale = size / min(height, width)
        if height < width:
            newh, neww = size, scale * width
        else:
            newh, neww = scale * height, size

        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        return neww, newh
