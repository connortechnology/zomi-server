from __future__ import annotations

import time
from functools import lru_cache
import warnings
from logging import getLogger
from typing import List, Optional, TYPE_CHECKING
from ...utils import resize_cv2_image


# from PIL import Image
import numpy as np

try:
    import cv2
except ImportError:
    warnings.warn("OpenCV not installed, please install OpenCV!", ImportWarning)
    cv2 = None
    raise
try:
    import netint.network
except ImportError:
    warnings.warn(
        "netint not installed, this is ok if you do not plan to use Quadra as detection processor. "
        "If you intend to use a Quadra please install the TPU libs and netint!",
        ImportWarning,
    )
    netint = None
    make_interpreter = None

from ...app import get_global_config
from ...Models.Enums import ModelType, ModelProcessor
from ...app import SERVER_LOGGER_NAME
from ...Models.config import DetectionResults, Result

if TYPE_CHECKING:
    from ...Models.config import QuadraModelConfig, QuadraModelOptions

logger = getLogger(SERVER_LOGGER_NAME)
LP: str = "NETINT:"

def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    #y[..., 0] = x[..., 0] - x[..., 2] / 2
    #y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2]
    y[..., 3] = x[..., 1] + x[..., 3]
    return y

def rescale_boxes(boxes, src_width, src_height, dst_width, dst_height):
    """Rescale boxes to original image dimensions"""
    input_shape = np.array(
        [src_width, src_height, dst_width, dst_height]
    )

    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    logger.debug(boxes)

    boxes *= np.array(
        [dst_width, dst_height, dst_width, dst_height]
    )
    logger.debug(boxes)
    boxes = np.rint(boxes).astype(int)
    logger.debug(boxes)
    return boxes

def percentages_to_coordinates(box, width, height):
    box[..., 0] *= width
    box[..., 1] *= height
    box[..., 2] *= width
    box[..., 3] *= height
    return box

def remove_padding(box, padding_width_percent, padding_height_percent):
    box[..., 0] += padding_width_percent
    box[..., 1] += padding_height_percent
    box[..., 2] += padding_width_percent
    box[..., 3] += padding_height_percent
    return box


class NETINTDetector:
    _classes: Optional[List] = None

    @property
    @lru_cache
    def classes(self):
        if self._classes is None:
            with open(self.config.classes) as f:
                self._classes = f.read().splitlines()
                logger.debug(self._classes)
        return self._classes

    @classes.setter
    def classes(self, value: List):
        self._classes = value

    def warm_up(self):
        """Warm up the Quadra by running a dummy inference."""
        lp = f"{LP}{self.name}:"
        runs = 2
        if self.model:
            logger.debug(f"{lp} warming up the model with {runs} iterations...")
            # create a random image to warm up Quadra
            #_h, _w = self.config.height, self.config.width
            #image = np.random.randint(0, 256, (_h, _w, 3), dtype=np.uint8)
            warmup_start = time.time()
            #for _ in range(runs):
                #_ = self.detect(image)
            logger.debug(f"perf:{lp} model warmed up in {time.time() - warmup_start:.5f} seconds")

    def __init__(self, model_config: QuadraModelConfig):
        global LP

        if netint is None:
            raise ImportError(
                f"{LP} netint is not installed, cannot use Quadra detectors"
            )

        # Model init params
        self.config: QuadraModelConfig = model_config
        self.options: QuadraModelOptions = self.config.detection_options
        self.processor: ModelProcessor = self.config.processor
        self.name: str = self.config.name
        self.model: Optional[Interpreter] = None
        if self.config.type_of == ModelType.FACE:
            logger.debug(
                f"{LP} ModelType=Face, this is for identification purposes only"
            )
            LP = f"{LP}Face:"
        self.load_model()
        #if (self.options.classes):


    def load_model(self):
        logger.debug(
            f"{LP} loading model {self.config.input.as_posix()}into {self.processor.upper()} processor memory: {self.name} ({self.config.id})"
        )
        t = time.time()
        try:
            self.model: Interpreter = netint.network.Interpreter(model_path=self.config.input.as_posix(), dev_id=self.config.device_id)
            self.input_details = self.model.get_input_details()
            logger.debug(self.input_details);
            self.output_details = self.model.get_output_details()
            logger.debug(self.output_details);

            if len(self.input_details) == 0 or len(self.output_details) == 0:
                raise RuntimeError(f"{LP} invalid network")

        except Exception as ex:
            ex = repr(ex)
            logger.error(
                f"{LP} failed to load model: {ex}"
            )
            words = ex.split(" ")
            for word in words:
                logger.info(
                        f"{LP} Quadra error detected. It could be a bad cable, needing to unplug/replug "
                        f"in the device or a system reboot."
                        )
        else:
            logger.debug(f"perf:{LP} loading took: {time.time() - t:.5f}s")


    def square_image(self, frame: np.ndarray):
        """Zero pad the matrix to make the image squared"""
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        logger.debug(
            f"{LP}squaring image-> '{self.name}' before padding: {frame.shape} - after padding: {result.shape}"
        )
        return result

    def pre_process(self, image, scale):
        #image = self.square_image(image)
        image = np.transpose(image,(2,0,1))
        mean = np.array([0.0,0.0,0.0])
        image = np.float32(image)
        image = image - mean[:,np.newaxis,np.newaxis]
        image = image * scale
        return np.float32(image)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def process(self, input, mask, anchors):

        anchors = [anchors[i] for i in mask]
        grid_h, grid_w = map(int, input.shape[0:2])

        box_confidence = self.sigmoid(input[..., 4])
        box_confidence = np.expand_dims(box_confidence, axis=-1)

        box_class_probs = self.sigmoid(input[..., 5:])

        box_xy = self.sigmoid(input[..., :2])
        box_wh = np.exp(input[..., 2:4])
        box_wh = box_wh * anchors

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid
        box_xy /= (grid_w, grid_h)
        box_wh /= (self.config.width, self.config.height)
        box_xy -= (box_wh / 2.)
        box = np.concatenate((box_xy, box_wh), axis=-1)

        return box, box_confidence, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with object threshold.

        # Arguments
            boxes: ndarray, boxes of objects.
            box_confidences: ndarray, confidences of objects.
            box_class_probs: ndarray, class_probs of objects.

        # Returns
            boxes: ndarray, filtered boxes.
            classes: ndarray, classes for boxes.
            scores: ndarray, scores for boxes.
        """
        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= self.config.detection_options.confidence)

        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]

        return boxes, classes, scores


    def nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes.

        # Arguments
            boxes: ndarray, boxes of objects.
            scores: ndarray, scores of objects.

        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.options.nms.threshold)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    def post_process(self, input_data):
        masks = [[3, 4, 5], [1, 2, 3]]
        anchors = [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]]

        boxes, classes, scores = [], [], []
        for input,mask in zip(input_data, masks):
            b, c, s = self.process(input, mask, anchors)
            b, c, s = self.filter_boxes(b, c, s)
            boxes.append(b)
            classes.append(c)
            scores.append(s)

        boxes = np.concatenate(boxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = self.nms_boxes(b, s)

            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        if not nclasses and not nscores:
            return boxes, classes, scores

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    async def detect(self, input_image: np.ndarray):
        """Performs object detection on the input image."""
        conf_threshold = self.config.detection_options.confidence
        nms = self.options.nms
        nms_str = f" - nms: {nms.threshold}" if nms.enabled else ""
        if not self.model:
            logger.warning(f"{LP} model not loaded? loading now...")
            self.load_model()

        if not self.model:
            raise RuntimeError(f"{LP} can't load model")

        input_height, input_width = input_image.shape[:2]
        logger.debug(input_image.shape)
        t = time.time()
        #cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        #factor = min(self.config.width / image_width, self.config.height / image_height)

        scale = 0.00392157
        logger.debug(
            f"{LP}detect: input image {input_width}*{input_height} - confidence: {conf_threshold}{nms_str}"
        )
        input_image = resize_cv2_image(input_image, self.config.width)
        scaled_height, scaled_width = input_image.shape[:2];
        input_image = self.square_image(input_image)
        padded_height, padded_width = input_image.shape[:2]
        padding_width_percent = 1-(scaled_width / padded_width);
        padding_height_percent = 1-(scaled_height / padded_height);
        logger.debug(f"Padding percentages: {padding_width_percent} = {padded_width} / {scaled_width}, {padding_height_percent} = {padded_height} / {scaled_height}");

        detect_height, detect_width = input_image.shape[:2]
        x_factor = detect_width / input_width
        y_factor = detect_height / input_height
  
        input_image = self.pre_process(input_image, scale)
        logger.debug(
            f"{LP}detect: input image {detect_width}*{detect_height} scale {scale} - confidence: {conf_threshold}{nms_str}"
        )
        try:

            #
            #async with get_global_config().async_locks.get(self.processor):
                # input format is image
            self.model.set_tensor(0, input_image)
            self.model.invoke()
        except Exception as ex:
            logger.error(f"{LP} Quadra error while calling invoke(): {ex}")
            raise ex
        else:

            b_boxes, labels, confs = [], [], []

            data_list = []
            for output_index in range(len(self.output_details)):
                tensor_output = self.model.get_tensor(output_index)
                data_list.append(tensor_output)

            NUM_CLS = len(self.classes)
            logger.debug(NUM_CLS)
            tensor0 = data_list[0].reshape(1,255,13,13)
            tensor0 = tensor0.reshape(-1, 5+NUM_CLS,tensor0.shape[2], tensor0.shape[3])
            tensor0 = np.transpose(tensor0, (2, 3, 0, 1))

            tensor1 = data_list[1].reshape(1,255,26,26)
            tensor1 = tensor1.reshape(-1,5+NUM_CLS,tensor1.shape[2],tensor1.shape[3])
            tensor1 = np.transpose(tensor1, (2, 3, 0, 1))

            input_data = []
            input_data.append(tensor0)
            input_data.append(tensor1)

            boxes, classes, scores = self.post_process(input_data)
            logger.debug(boxes);
            boxes = remove_padding(boxes, padding_width_percent, padding_height_percent)
            logger.debug('after remove padding')
            logger.debug(boxes);
            boxes = percentages_to_coordinates(boxes, scaled_width, scaled_height)

            logger.debug('to coordinates');
            logger.debug(boxes);
            logger.debug('to xyxy');
            boxes = xywh2xyxy(boxes)
            logger.debug(boxes);
            boxes = rescale_boxes(boxes, scaled_width, scaled_height, input_width, input_height)
            logger.debug('after rescale');
            logger.debug(boxes);

        result = DetectionResults(
            success=True if classes.size else False,
            type=self.config.type_of,
            processor=self.processor,
            name=self.name,
            results=[
                Result(
                    label=self.classes[classes[i]],
                    confidence=scores[i],
                    bounding_box=boxes[i],
                )
                for i in range(len(classes))
            ],
        )
        return result
