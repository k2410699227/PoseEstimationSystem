import io

import numpy as np
from PIL import Image
from detectron2.utils.visualizer import Visualizer, _create_text_labels


def bytes_to_BGRImg(stream):
    with Image.open(io.BytesIO(stream)) as img:
        image = np.asarray(img)
        image = image[:, :, ::-1]
        return image


class MyVisualizer(Visualizer):
    def draw_instance_predictions(self, predictions, random_color=True):
        """
                Draw instance-level prediction results on an image.

                Args:
                    predictions (Instances): the output of an instance detection/segmentation
                        model. Following fields will be used to draw:
                        "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

                Returns:
                    output (VisImage): image object with visualizations.
                """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        num_instances = len(boxes)
        masks = None
        default_color = [0, 0.6, 0]
        if random_color:
            colors = None
        else:
            colors = [default_color for _ in range(num_instances)]
        alpha = 0.5

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output
