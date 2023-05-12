from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.defaults import DefaultPredictor
import detectron2.data.transforms as T


class MyPredictor(DefaultPredictor):
    def __init__(self, model):
        # self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = model
        self.model.eval()

        # if len(cfg.DATASETS.TEST):
        #     self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        #
        # checkpointer = DetectionCheckpointer(self.model)
        # checkpointer.load("fcpose50.pth")
        # self.model.eval()
        self.aug = T.ResizeShortestEdge(
            [512, 512], 1333
        )

        self.input_format = "BGR"
        assert self.input_format in ["RGB", "BGR"], self.input_format
