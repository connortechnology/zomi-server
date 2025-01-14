import logging
from enum import Enum

from ..Log import SERVER_LOGGER_NAME

logger = logging.getLogger(SERVER_LOGGER_NAME)


class ModelType(str, Enum):
    OBJECT = "object"
    FACE = "face"
    ALPR = "alpr"

    def __repr__(self):
        return f"<{self.__class__.__name__}: {str(self.name).lower()} detection>"

    def __str__(self):
        return self.__repr__()


class OpenCVSubFrameWork(str, Enum):
    DARKNET = "darknet"
    CAFFE = "caffe"
    TENSORFLOW = "tensorflow"
    TORCH = "torch"
    VINO = "vino"
    ONNX = "onnx"
    NETINT = "netint"

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"

    def __str__(self):
        return f"{self.name}"


class HTTPSubFrameWork(str, Enum):
    NONE = "none"
    REKOGNITION = "rekognition"


class ALPRSubFrameWork(str, Enum):
    OPENALPR = "openalpr"
    PLATE_RECOGNIZER = "plate_recognizer"
    REKOR = "rekor"

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"

    def __str__(self):
        return f"{self.name}"


class ModelFrameWork(str, Enum):
    OPENCV = "opencv"
    HTTP = "http"
    CORAL = "coral"
    TORCH = "torch"
    DEEPFACE = "deepface"
    ALPR = "alpr"
    FACE_RECOGNITION = "face_recognition"
    ORT = "ort"
    TRT = "trt"
    NETINT = "netint"


class ModelProcessor(str, Enum):
    NONE = "none"
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    QUADRA = "quadra"

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"

    def __str__(self):
        return f"{self.name}"


class FaceRecognitionLibModelTypes(str, Enum):
    CNN = "cnn"
    HOG = "hog"


class ALPRAPIType(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


class ALPRService(str, Enum):
    OPENALPR = "openalpr"
    PLATE_RECOGNIZER = "plate_recognizer"
    SCOUT = OPENALPR
