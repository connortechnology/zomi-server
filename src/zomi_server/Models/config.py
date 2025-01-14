import enum
import json
import logging
import time
import uuid
from asyncio import BoundedSemaphore
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple, Annotated, Any

import numpy as np
import yaml
from pydantic import (
    BaseModel,
    Field,
    FieldValidationInfo,
    field_validator,
    model_validator,
    IPvAnyAddress,
    PositiveInt,
    SecretStr,
    IPvAnyNetwork,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from .Enums import (
    ModelType,
    ModelFrameWork,
    ModelProcessor,
    FaceRecognitionLibModelTypes,
    ALPRAPIType,
    ALPRService,
    HTTPSubFrameWork,
    OpenCVSubFrameWork,
    ALPRSubFrameWork,
)
from .validators import (
    str2path,
    validate_enabled,
    validate_not_enabled,
    validate_log_level,
    validate_model_labels,
)
from .DEFAULTS import *
from ..Log import SERVER_LOGGER_NAME
from ..ML.Detectors.color_detector import ColorDetector

logger = logging.getLogger(SERVER_LOGGER_NAME)


class NMSOptions(BaseModel):
    enabled: bool = Field(True, description="Enable Non-Maximum Suppression")
    threshold: Optional[float] = Field(
        0.35, ge=0.0, le=1.0, description="Non-Maximum Suppression Threshold"
    )


class OutputType(str, enum.Enum):
    yolov3 = "yolov3"
    yolov4 = "yolov4"
    yolov5 = "yolov5"
    yolov6 = "yolov6"
    yolov7 = "yolov7"
    yolov8 = "yolov8"
    yolov9 = "yolov9"
    yolonas = "yolonas"
    yolov10 = "yolov10"


class Testing(BaseModel):
    enabled: bool = Field(False)
    substitutions: Dict[str, str] = Field(default_factory=dict)


class DefaultEnabled(BaseModel):
    enabled: bool = Field(True)

    _v = field_validator("enabled", mode="before")(validate_enabled)


class DefaultNotEnabled(BaseModel):
    enabled: bool = Field(False)

    _v = field_validator("enabled", mode="before")(validate_not_enabled)


class LoggingLevelBase(BaseModel):
    level: Optional[int] = None

    _validate_log_level = field_validator("level", mode="before")(validate_log_level)


class LoggingSettings(LoggingLevelBase):
    class ConsoleLogging(DefaultEnabled, LoggingLevelBase):
        pass

    class SyslogLogging(DefaultNotEnabled, LoggingLevelBase):
        address: Optional[str] = Field("")

    class FileLogging(DefaultEnabled, LoggingLevelBase):
        path: Path = Field("/var/log/zm")
        filename_prefix: str = Field("zomi_server")
        file_name: Optional[str] = None
        user: str = Field(default="www-data")
        group: str = Field(default="www-data")

        _validate_path = field_validator("path", mode="before")(str2path)

    class SanitizeLogging(DefaultNotEnabled):
        replacement_str: str = Field(default="<sanitized>")

    class IntegrateZMLogging(DefaultNotEnabled):
        debug_level: int = Field(default=4)

    level: int = logging.INFO
    console: ConsoleLogging = Field(default_factory=ConsoleLogging)
    syslog: SyslogLogging = Field(default_factory=SyslogLogging)
    integrate_zm: IntegrateZMLogging = Field(default_factory=IntegrateZMLogging)
    file: FileLogging = Field(default_factory=FileLogging)
    sanitize: SanitizeLogging = Field(default_factory=SanitizeLogging)


class Result(BaseModel):
    label: str
    confidence: float
    bounding_box: List[int]
    color: Optional[Dict[str, float]] = None

    def __eq__(self, other):
        if not isinstance(other, Result):
            return False
        return (
            self.label == other.label
            and self.confidence == other.confidence
            and self.bounding_box == other.bounding_box
        )

    def __str__(self):
        _c: Optional[str] = None
        if self.color:
            _c = f" [Color: {self.color}]"
        return f"'{self.label}'{_c} ({self.confidence:.2f}) @ {self.bounding_box}"

    def __repr__(self):
        _c: Optional[str] = None
        if self.color:
            _c = f" [Color: {self.color}]"
        return (
            f"<'{self.label}'{_c} ({self.confidence * 100:.2f}%) @ {self.bounding_box}>"
        )


class DetectionResults(BaseModel, arbitrary_types_allowed=True):
    success: bool = Field(...)
    name: str = Field(...)
    type: ModelType = Field(...)
    processor: ModelProcessor = Field(...)
    results: Optional[List[Result]] = Field(None)
    # extra_image_data: Optional[Dict[str, Any]] = Field(None, repr=False)

    def get_labels(self) -> list[Any] | tuple[list[str], list[float], list[list[int]]]:
        if not self.results or self.results is None:
            return []
        return (
            [r.label for r in self.results],
            [r.confidence for r in self.results],
            [r.bounding_box for r in self.results],
        )


class OCRConfig(BaseModel):
    class TessConfig(BaseModel):
        enabled: bool = Field(
            False, description="Enable Tesseract OCR on cropped bounding boxes"
        )
        lang: Optional[str] = Field(
            "eng", description="List of languages to use for OCR, see Tesseract docs"
        )
        executable: Optional[Path] = Field(
            None,
            description="Path to the Tesseract executable, if None, will use 'which tesseract' to find it",
        )
        oem: Annotated[Union[None, int], Field(ge=0, le=3)] = None
        img_psm: Annotated[int, Field(ge=0, le=13), None] = None
        char_psm: Annotated[int, Field(ge=0, le=13), None] = None

    class PaddleOCRConfig(BaseModel):
        enabled: bool = Field(
            False, description="Enable PaddleOCR on cropped bounding boxes"
        )

    class EasyOCRConfig(BaseModel):
        enabled: bool = Field(
            False, description="Enable EasyOCR on cropped bounding boxes"
        )
        lang: List[str] = Field(
            ["en"], description="List of languages to use for OCR, see EasyOCR docs"
        )
        gpu: bool = Field(
            False, description="Enable GPU acceleration (if available) for EasyOCR"
        )
        width_ths: Annotated[
            float,
            Field(
                0.5,
                le=1.0,
                ge=0,
                description="Width threshold for Bounding Box Merging, see EasyOCR docs",
            ),
        ]
        height_ths: Annotated[
            float,
            Field(
                0.5,
                description="Height threshold for Bounding Box Merging, see EasyOCR docs",
            ),
        ]
        slope_ths: Annotated[
            float,
            Field(
                0.5,
                description="Slope threshold for Bounding Box Merging, see EasyOCR docs",
            ),
        ]
        ycenter_ths: Annotated[
            float,
            Field(
                0.5,
                description="Y Center threshold for Bounding Box Merging, see EasyOCR docs",
            ),
        ]
        min_size: Annotated[
            int,
            Field(20, description="Minimum size of a textbox in px, see EasyOCR docs"),
        ]
        contrast_ths: Annotated[
            float,
            Field(
                0.1,
                description="Contrast threshold for image binarization, see EasyOCR docs",
            ),
        ]
        adjust_contrast: Annotated[
            float,
            Field(
                0.5,
                description="Adjust contrast for image binarization, see EasyOCR docs",
            ),
        ]

    class OCRDarkConfig(BaseModel):
        enabled: Optional[bool] = Field(True)
        dark_thresh: Annotated[int, Field(ge=0, le=255), None] = None
        mean_intensity: Annotated[int, Field(ge=0, le=255), None] = None
        contrast: Annotated[float, Field(ge=0, le=255), None] = None

    easy_ocr: EasyOCRConfig = Field(default_factory=EasyOCRConfig)
    paddle_ocr: PaddleOCRConfig = Field(default_factory=PaddleOCRConfig)
    tesseract: Optional[TessConfig] = Field(default_factory=TessConfig)

    darkness: OCRDarkConfig = Field(default_factory=OCRDarkConfig)

    debug_images: Optional[bool] = Field(False)
    dbg_img_dir: Optional[Path] = None

    word_similarity_thresh: Optional[int] = None

    blur_kernel: Optional[Tuple[int, int]] = None
    dilate_kernel: Optional[Tuple[int, int]] = None
    erode_kernel: Optional[Tuple[int, int]] = None

    crop_leeway_perc: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = None

    # For edge thresh index 0 is lower and 1 is higher
    # edges with a thresh below lower are discarded
    # edges with a thresh above higher are kept and considered STRONG edges
    # edges with a thresh between lower and higher are considered WEAK edges ->
    # weak edges are kept if they are connected to a strong edge

    lp_edge_thresh: Optional[Tuple[int, int]] = None
    char_edge_thresh: Optional[Tuple[int, int]] = None

    confidence: Annotated[Union[None, float], Field(ge=0.0, le=1.0, default=0.4)] = None

    class CharExtractConfig(BaseModel):
        enabled: Optional[bool] = Field(True)
        right2left: Optional[bool] = Field(
            False, description="Extract characters from right to left"
        )

        min_area_perc: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = Field(
            0.02, description="Percentage of image area the rectangle can be"
        )
        max_area_perc: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = Field(
            0.6, description="Percentage of image area the rectangle can be"
        )
        max_w_perc: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = Field(
            0.3875, description="Percent of the images width that the char can be"
        )
        min_w_perc: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = Field(
            0.025, description="Percent of the images width that the char can be"
        )
        max_h_perc: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = Field(
            0.9, description="Percent of image height the char can be"
        )
        min_h_perc: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = Field(
            0.2, description="Percent of image height the char can be"
        )

        # 1% of the total image pixels will be added to the character crop in w & h
        # Every 100 pixels, 1 pixel will be added to the crop
        leeway_w_perc: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = Field(
            0.01,
            description="When cropping image add this percent of the total image in pixels to the cropped image",
        )
        leeway_h_perc: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = Field(
            0.01,
            description="When cropping image add this percent of the total image in pixels to the cropped image",
        )
        min_aspect_ratio: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = Field(
            0.1, description="Minimum aspect ratio of the character"
        )
        max_aspect_ratio: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = Field(
            0.9, description="Maximum aspect ratio of the character"
        )

        contour_approx_scale: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = (
            Field(0.02, description="Approximation accuracy for the contour")
        )

    char_extract: Optional[CharExtractConfig] = Field(default_factory=CharExtractConfig)

    class PlatePerspectiveTransformConfig(BaseModel):
        enabled: Optional[bool] = Field(False)
        min_area_perc: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = Field(
            0.02, description="Percentage of image area the rectangle can be"
        )
        max_area_perc: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = Field(
            0.6, description="Percentage of image area the rectangle can be"
        )
        max_w_perc: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = Field(
            0.3875, description="Percent of the images width that the char can be"
        )
        min_w_perc: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = Field(
            0.025, description="Percent of the images width that the char can be"
        )
        max_h_perc: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = Field(
            0.9, description="Percent of image height the char can be"
        )
        min_h_perc: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = Field(
            0.2, description="Percent of image height the char can be"
        )
        max_aspect_ratio: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = Field(
            0.9, description="Maximum aspect ratio of the character"
        )
        min_aspect_ratio: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = Field(
            0.1, description="Minimum aspect ratio of the character"
        )
        contour_approx_scale: Annotated[Union[None, float], Field(ge=0.0, le=1.0)] = (
            Field(0.02, description="Approximation accuracy for the contour")
        )

    warp_plate: Optional[PlatePerspectiveTransformConfig] = Field(
        default_factory=PlatePerspectiveTransformConfig
    )

    # plate_recognizer:


class LockSetting(BaseModel):
    """Default file lock options"""

    max: int = Field(
        4, ge=1, le=100, description="Maximum number of parallel processes"
    )
    timeout: int = Field(
        30, ge=1, le=480, description="Timeout in seconds for acquiring a lock"
    )


class LockSettings(DefaultEnabled):
    gpu: LockSetting = Field(
        default_factory=LockSetting, description="GPU Lock Settings"
    )
    cpu: LockSetting = Field(
        default_factory=LockSetting, description="CPU Lock Settings"
    )
    tpu: LockSetting = Field(
        default_factory=LockSetting, description="TPU Lock Settings"
    )

    def get(self, device: str) -> LockSetting:
        if device:
            device = device.casefold()
            if device == "gpu":
                return self.gpu
            elif device == "cpu":
                return self.cpu
            elif device == "tpu":
                return self.tpu
            else:
                raise ValueError(f"Invalid device type: {device}")
        else:
            raise ValueError("Device type must be specified")


class ServerSettings(BaseModel):
    class AuthSettings(DefaultEnabled):
        db_file: Path = Field(..., description="TinyDB user DB file")
        expire_after: int = Field(
            60, ge=1, description="JWT Expiration time in minutes"
        )
        sign_key: Optional[SecretStr] = Field(None, description="JWT Sign Key")
        algorithm: str = Field("HS256", description="JWT Algorithm")

    address: Union[IPvAnyAddress] = Field(
        "0.0.0.0", description="Server listen address"
    )

    port: PositiveInt = Field(5000, description="Server listen port")
    debug: bool = Field(
        default=False, description="Uvicorn debug mode - For development only"
    )
    auth: AuthSettings = Field(default_factory=AuthSettings, description="JWT Settings")


class BaseModelOptions(BaseModel):
    confidence: Optional[float] = Field(
        0.2, ge=0.0, le=1.0, description="Confidence Threshold"
    )


class ORTModelOptions(BaseModelOptions):
    nms: NMSOptions = Field(default_factory=NMSOptions, description="NMS Options")


class TRTModelOptions(BaseModelOptions):
    nms: NMSOptions = Field(default_factory=NMSOptions, description="NMS Options")


class TPUModelOptions(BaseModelOptions, extra="allow"):
    nms: NMSOptions = Field(default_factory=NMSOptions, description="NMS Options")


class TorchModelOptions(BaseModelOptions):
    nms: NMSOptions = Field(default_factory=NMSOptions, description="NMS Options")

class NETINTModelOptions(BaseModelOptions):
    nms: NMSOptions = Field(default_factory=NMSOptions, description="NMS Options")

class CV2YOLOModelOptions(BaseModelOptions):
    nms: Optional[float] = Field(
        0.4, ge=0.0, le=1.0, description="Non-Maximum Suppression Threshold"
    )


class FaceRecognitionLibModelDetectionOptions(BaseModelOptions):
    # face_recognition lib config Options
    model: Optional[FaceRecognitionLibModelTypes] = Field(
        FaceRecognitionLibModelTypes.CNN,
        examples=["hog", "cnn"],
        description="Face Detection Model to use. 'cnn' is more accurate but slower on CPUs. "
        "'hog' is faster but less accurate",
    )
    upsample_times: Optional[int] = Field(
        1,
        ge=0,
        description="How many times to upsample the image looking for faces. "
        "Higher numbers find smaller faces but take longer.",
    )
    num_jitters: Optional[int] = Field(
        1,
        ge=0,
        description="How many times to re-sample the face when calculating encoding. "
        "Higher is more accurate, but slower (i.e. 100 is 100x slower)",
    )

    max_size: Optional[int] = Field(
        600,
        ge=100,
        description="Maximum size (Width) of image to load into memory for "
        "face detection (image will be scaled)",
    )
    recognition_threshold: Optional[float] = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="Recognition distance threshold for face recognition",
    )


class FaceRecognitionLibModelTrainingOptions(BaseModelOptions):
    model: Optional[FaceRecognitionLibModelTypes] = Field(
        FaceRecognitionLibModelTypes.CNN,
        examples=["hog", "cnn"],
        description="Face Detection Model to use. 'cnn' is more accurate but slower on CPUs. "
        "'hog' is faster but less accurate",
    )
    upsample_times: Optional[int] = Field(
        1,
        ge=0,
        description="How many times to upsample the image while detecting faces. "
        "Higher numbers find smaller faces but take longer.",
    )
    num_jitters: Optional[int] = Field(
        1,
        ge=0,
        description="How many times to re-sample the face when calculating encoding. "
        "Higher is more accurate, but slower (i.e. 100 is 100x slower)",
    )

    max_size: Optional[int] = Field(
        600,
        ge=50,
        description="Maximum size (Width) of image to load into memory for "
        "face detection (image will be scaled)",
    )
    dir: Optional[Path] = Field(
        None,
        description="Directory to load training images from. If None, the default directory will be used",
    )


class ALPRModelOptions(BaseModelOptions):
    max_size: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum size (Width) of image to load into memory",
    )


class OpenALPRLocalModelOptions(ALPRModelOptions):
    binary_path: Optional[str] = Field(
        "alpr", description="OpenALPR binary name or absolute path"
    )
    binary_params: Optional[str] = Field(
        "-d",
        description="OpenALPR binary parameters (-j is ALWAYS passed)",
        example="-p ca -c US",
    )


class OpenALPRCloudModelOptions(ALPRModelOptions):
    # For an explanation of params, see http://doc.openalpr.com/api/?api=cloudapi
    recognize_vehicle: Optional[bool] = Field(
        True,
        description="If True, will attempt to recognize the vehicle type (ie: Ford Mustang)",
    )
    country: Optional[str] = Field(
        "us",
        description="Country of license plate to recognize.",
    )
    state: Optional[str] = Field(
        None,
        description="State of license plate to recognize.",
    )


class PlateRecognizerModelOptions(BaseModelOptions):
    stats: Optional[bool] = Field(
        False,
        description="Return stats about the plate recognition request",
    )
    payload: Optional[Dict] = Field(
        default_factory=dict,
        description="Override the payload sent to the Plate Recognizer API, must be a JSON serializable string",
    )
    config: Optional[Dict] = Field(
        None,
        description="Override the config sent to the Plate Recognizer API, must be a JSON serializable string",
    )
    # If you want to specify regions. See http://docs.platerecognizer.com/#regions-supported
    regions: List[str] = Field(
        None,
        example=["us", "cn", "kr", "ca"],
        description="List of regions to search for plates in. If not specified, "
        "all regions are searched. See http://docs.platerecognizer.com/#regions-supported",
    )
    # minimal confidence for actually detecting a plate (the presence of a license plate, not the actual plate text)
    min_dscore: float = Field(
        0.1,
        ge=0,
        le=1.0,
        description="Minimal confidence for detecting a license plate in the image",
    )
    # minimal confidence for the translated text (plate number)
    min_score: float = Field(
        0.5,
        ge=0,
        le=1.0,
        description="Minimal confidence for the translated text from the plate",
    )
    max_size: int = Field(
        1600,
        ge=1,
        description="Maximum size (Width) of image",
    )

    @field_validator("config", "payload")
    @classmethod
    def check_json_serializable(cls, v):
        try:
            json.dumps(v)
        except TypeError:
            raise ValueError("Must be JSON serializable, check formatting...")
        return v


class BaseModelConfig(BaseModel):
    """This is the base model config that all models inherit from."""

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4, description="Unique ID of the model", init=False
    )
    name: str = Field(..., description="model name")
    enabled: bool = Field(True, description="model enabled")
    description: str = Field(None, description="model description")
    # todo: add validator that detects framework and sets a default sub framework if it is None.
    framework: ModelFrameWork = Field(
        default=ModelFrameWork.ORT, description="model framework"
    )
    sub_framework: Optional[
        Union[
            OpenCVSubFrameWork,
            HTTPSubFrameWork,
            ALPRSubFrameWork,
        ]
    ] = Field(OpenCVSubFrameWork.DARKNET, description="sub-framework to use for model")
    type_of: ModelType = Field(
        ModelType.OBJECT,
        description="model type (object, face, alpr)",
        alias="model_type",
    )
    processor: Optional[ModelProcessor] = Field(
        ModelProcessor.CPU, description="Processor to use for model"
    )


    detection_options: Union[
        BaseModelOptions,
        FaceRecognitionLibModelDetectionOptions,
        OpenALPRLocalModelOptions,
        OpenALPRCloudModelOptions,
        PlateRecognizerModelOptions,
        ALPRModelOptions,
        TPUModelOptions,
        TorchModelOptions,
        TRTModelOptions,
        ORTModelOptions,
        None,
    ] = Field(
        default_factory=BaseModelOptions,
        description="Default Configuration for the model",
    )
    ocr: Optional[OCRConfig] = Field(
        default_factory=OCRConfig,
        description="Optical Character Recognition (OCR) for the cropped bounding boxes of the model",
    )
    detect_color: Optional[bool] = Field(
        False,
        description="Enable Color Detection for the cropped bounding boxes of the model",
    )

    labels: List[str] = Field(
        default=None,
        description="model labels parsed into a list of strings",
        repr=False,
        exclude=True,
    )

    @field_validator("name")
    @classmethod
    def check_name(cls, v):
        v = str(v).strip().casefold()
        return v


class TPUModelConfig(BaseModelConfig):
    input: Optional[Path] = Field(None, description="model file/dir path (Optional)")
    config: Optional[Path] = Field(
        None, description="model config file path (Optional)"
    )
    classes: Optional[Path] = Field(
        None, description="model classes file path (Optional)"
    )
    height: Optional[int] = Field(
        416, ge=1, description="Model input height (resized for model)"
    )
    width: Optional[int] = Field(
        416, ge=1, description="Model input width (resized for model)"
    )
    square: Optional[bool] = Field(
        False, description="Zero pad the image to be a square; 1920x1080 = 1920x1920"
    )

    _validate_labels = field_validator("labels", check_fields=False)(validate_model_labels)

    @field_validator("config", "input", "classes", mode="before")
    @classmethod
    def str_to_path(cls, v, info: FieldValidationInfo) -> Optional[Path]:
        msg = f"{info.field_name} must be a path or a string of a path"
        model_name = info.data.get("name", "Unknown Model")
        model_input: Optional[Path] = info.data.get("input")
        lp = f"Model Name: {model_name} ->"

        if v is None:
            return v
        elif not isinstance(v, (Path, str)):
            raise ValueError(msg)
        elif isinstance(v, str):
            v = Path(v)
        if info.field_name == "config":
            if model_input and model_input.suffix == ".weights":
                msg = f"'{info.field_name}' is required when 'input' is a DarkNet .weights file"
        return v

class NETINTModelConfig(BaseModelConfig):
    input: Optional[Path] = Field(None, description="model file/dir path (Optional)")
    config: Optional[Path] = Field(
        None, description="model config file path (Optional)"
    )
    classes: Optional[Path] = Field(
        None, description="model classes file path (Optional)"
    )
    device_id: Optional[int] = Field(
        0, ge=0, le=4, description="Device ID"
    )
    height: Optional[int] = Field(
        416, ge=1, description="Model input height (resized for model)"
    )
    width: Optional[int] = Field(
        416, ge=1, description="Model input width (resized for model)"
    )
    square: Optional[bool] = Field(
        False, description="Zero pad the image to be a square; 1920x1080 = 1920x1920"
    )

    _validate_labels = field_validator("labels", check_fields=False)(validate_model_labels)

    @field_validator("config", "input", "classes", mode="before")
    @classmethod
    def str_to_path(cls, v, info: FieldValidationInfo) -> Optional[Path]:
        msg = f"{info.field_name} must be a path or a string of a path"
        model_name = info.data.get("name", "Unknown Model")
        model_input: Optional[Path] = info.data.get("input")
        lp = f"Model Name: {model_name} ->"

        if v is None:
            return v
        elif not isinstance(v, (Path, str)):
            raise ValueError(msg)
        elif isinstance(v, str):
            v = Path(v)
        return v

class ORTModelConfig(BaseModelConfig):
    input: Path = Field(None, description="model file/dir path (Optional)")
    classes: Path = Field(default=None, description="model labels file path (Optional)")
    height: Optional[int] = Field(
        416, ge=1, description="Model input height (resized for model)"
    )
    width: Optional[int] = Field(
        416, ge=1, description="Model input width (resized for model)"
    )
    square: Optional[bool] = Field(
        False, description="Zero pad the image to be a square"
    )
    output_type: Optional[OutputType] = None
    gpu_idx: Optional[int] = 0

    @model_validator(mode="after")
    def _validate_model(self):
        model_name = self.name
        model_input: Optional[Path] = self.input
        labels = self.labels
        self.labels = validate_model_labels(
            labels, info=None, model_name=model_name, labels_file=self.classes
        )
        v = model_input
        assert isinstance(v, (Path, str)), f"Invalid type: {type(v)} for {v}"
        if isinstance(v, str):
            v = Path(v)

        return self


class TRTModelConfig(BaseModelConfig):
    input: Path = Field(None, description="model file/dir path (Optional)")
    classes: Path = Field(default=None, description="model labels file path (Optional)")
    height: Optional[int] = Field(
        416, ge=1, description="Model input height (resized for model)"
    )
    width: Optional[int] = Field(
        416, ge=1, description="Model input width (resized for model)"
    )
    square: Optional[bool] = Field(
        False, description="Zero pad the image to be a square"
    )
    output_type: Optional[OutputType] = None
    gpu_idx: Optional[int] = 0

    @model_validator(mode="after")
    def _validate_model(self):
        model_name = self.name
        model_input: Optional[Path] = self.input
        labels = self.labels
        self.labels = validate_model_labels(
            labels, info=None, model_name=model_name, labels_file=self.classes
        )
        v = model_input
        assert isinstance(v, (Path, str)), f"Invalid type: {type(v)} for {v}"
        if isinstance(v, str):
            v = Path(v)

        return self


class CV2YOLOModelConfig(BaseModelConfig):

    input: Path = Field(None, description="model file/dir path (Optional)")
    classes: Path = Field(default=None, description="model labels file path (Optional)")
    config: Optional[Path] = Field(
        default=None, description="model config file path (Optional)"
    )
    height: Optional[int] = Field(
        416, ge=1, description="Model input height (resized for model)"
    )
    width: Optional[int] = Field(
        416, ge=1, description="Model input width (resized for model)"
    )
    square: Optional[bool] = Field(
        False, description="Zero pad the image to be a square"
    )
    cuda_fp_16: Optional[bool] = Field(
        False, description="Use Floating Point 16 Backend (CUDA - EXPERIMENTAL)"
    )
    gpu_idx: Optional[int] = Field(0, description="GPU Index to use")

    output_type: Optional[OutputType] = Field(None, description="Model output type")

    @model_validator(mode="after")
    def _validate_model(self):
        model_name = self.name
        model_input: Optional[Path] = self.input
        model_config: Optional[Path] = self.config or None
        labels = self.labels
        self.labels = validate_model_labels(
            labels, info=None, model_name=model_name, labels_file=self.classes
        )

        for v in (model_input, model_config):
            if not v:
                continue
            assert isinstance(v, (Path, str)), f"Invalid type: {type(v)} for {v}"
            if v == model_config:
                if v is None:
                    if model_input and model_input.suffix == ".weights":
                        raise ValueError(
                            f"'{v.__class__.name}' is required when 'input' is a DarkNet .weights file"
                        )
                    return v
            elif isinstance(v, str):
                v = Path(v)

        return self


class FaceRecognitionLibModelConfig(BaseModelConfig):
    """Config cant be changed after loading - Options can be changed"""

    class UnknownFaceSettings(BaseModel):
        enabled: Optional[bool] = Field(
            True,
            description="If True, unknown faces will be saved to disk for possible training",
        )
        label_as: Optional[str] = Field(
            "Unknown",
            description="Label to use for unknown faces. If 'enabled' is False, this is ignored",
        )
        dir: Optional[Path] = Field(
            None,
            description="Directory to save unknown faces to. If 'enabled' is False, this is ignored",
        )
        leeway_pixels: int = Field(
            0,
            description="Unknown faces leeway pixels, used when cropping the image to capture a face",
        )

    detection_options: FaceRecognitionLibModelDetectionOptions = Field(
        default_factory=FaceRecognitionLibModelDetectionOptions,
        description="Default Configuration for the model",
    )

    training_options: Optional[FaceRecognitionLibModelTrainingOptions] = Field(
        default_factory=FaceRecognitionLibModelTrainingOptions,
        description="Configuration for training faces",
    )

    unknown_faces: UnknownFaceSettings = Field(
        default_factory=UnknownFaceSettings,
        description="Settings for handling unknown faces",
    )


class ALPRModelConfig(BaseModelConfig):
    api_type: ALPRAPIType = Field(ALPRAPIType.LOCAL, description="ALPR Service Type")
    service: ALPRService = Field(
        ALPRService.OPENALPR, description="ALPR Service to use"
    )
    api_key: str = Field(None, description="ALPR API Key (Cloud/Local)")
    api_url: str = Field(None, description="ALPR API URL (Cloud/Local)")


class CV2HOGModelConfig(BaseModelConfig):
    stride: str = None
    padding: str = None
    scale: float = None
    mean_shift: bool = False


class RekognitionModelConfig(BaseModelConfig):
    aws_access_key_id: str = None
    aws_secret_access_key: str = None
    # aws_session_token: str = None
    # aws_profile: str = None
    region_name: str = None


class DeepFaceModelConfig(BaseModelConfig):
    pass


class TorchModelConfig(BaseModelConfig):
    class PreTrained(DefaultEnabled):
        name: Optional[str] = Field(
            "default",
            pattern=r"(accurate|fast|default|balanced|high_performance|low_performance)",
        )

        @field_validator("name", mode="before", check_fields=False)
        @classmethod
        def _validate_model_name(
            cls, v: Optional[str], info: FieldValidationInfo
        ) -> str:
            if v is None:
                v = "default"
            return v

    input: Optional[Path] = None
    classes: Optional[Path] = None
    num_classes: Optional[int] = None

    gpu_idx: Optional[int] = None

    pretrained: Optional[PreTrained] = Field(default_factory=PreTrained)

    _validate_labels = field_validator("labels", check_fields=False)(validate_model_labels)
    _validate = field_validator("input", "classes", mode="before")(str2path)


class ColorDetectSettings(DefaultEnabled):
    top_n: Optional[int] = Field(
        5, ge=1, description="Number of dominant colors to detect"
    )
    spec: Optional[Annotated[str, Field(pattern=r"(html4|css3|css2|css21)")]] = "html4"
    labels: Optional[List[str]] = Field(
        None,
        description="List of labels to run color detection on, leave empty to run on all detected objects",
    )

    @field_validator("spec", mode="after")
    def to_lower(cls, v):
        if v is not None and isinstance(v, str):
            if v:
                return v.lower()
        return v


def _replace_vars(search_str: str, var_pool: Dict) -> Dict:
    """Replace variables in a string.


    Args:
        search_str (str): String to search for variables '${VAR_NAME}'.
        var_pool (Dict): Dictionary of variables used to replace.

    """
    import re

    if var_list := re.findall(r"\$\{(\w+)\}", search_str):
        logger.debug(
            f"Found the following substitution variables: {list(set(var_list))}"
        )
        # substitute variables
        _known_vars = []
        _unknown_vars = []
        __seen_vars = []
        for var in var_list:
            if var in var_pool:
                # logger.debug(
                #     f"substitution variable '{var}' IS IN THE sub POOL! VALUE: "
                #     f"{var_pool[var]} [{type(var_pool[var])}]"
                # )
                value = var_pool[var]
                if value is None:
                    value = ""
                elif value is True:
                    value = "yes"
                elif value is False:
                    value = "no"
                search_str = search_str.replace(f"${{{var}}}", str(value))
                if var not in __seen_vars:
                    _known_vars.append(var)

            else:
                if var not in __seen_vars:
                    _unknown_vars.append(var)
            __seen_vars.append(var)

        if _unknown_vars:
            logger.warning(
                f"The following variables have no configured substitution value: {_unknown_vars}"
            )
        if _known_vars:
            logger.debug(
                f"The following variables have been substituted: {_known_vars}"
            )
    else:
        logger.debug(f"No substitution variables found.")
    return yaml.safe_load(search_str)


class SystemSettings(BaseModel):
    models_dir: Optional[Path] = Field(Path(DEF_SRV_SYS_MODELDIR), alias="model_dir")
    image_dir: Optional[Path] = Field(Path(DEF_SRV_SYS_IMAGEDIR))
    config_path: Optional[Path] = Field(Path(DEF_SRV_SYS_CONFDIR))
    variable_data_path: Optional[Path] = Field(DEF_SRV_SYS_DATADIR)
    tmp_path: Optional[Path] = Field(Path(DEF_SRV_SYS_TMPDIR))
    thread_workers: Optional[int] = Field(DEF_SRV_SYS_THREAD_WORKERS)


class UvicornSettings(BaseModel):
    debug: Optional[bool] = Field(False, description="Uvicorn debug mode")
    proxy_headers: Optional[bool] = False
    forwarded_allow_ips: Optional[List[Union[IPvAnyAddress, IPvAnyNetwork]]] = Field(
        default_factory=list
    )
    grab_cloudflare_ips: Optional[bool] = Field(False)
    reload: Optional[bool] = Field(False, description="Uvicorn reload")
    reload_dirs: Optional[List[Path]] = Field(
        default_factory=list, description="Uvicorn reload directories"
    )


class Settings(BaseModel, arbitrary_types_allowed=True):
    testing: Testing = Field(default_factory=Testing)
    substitutions: Dict = Field(default_factory=dict)
    uvicorn: UvicornSettings = Field(default_factory=UvicornSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    locks: LockSettings = Field(
        default_factory=LockSettings, description="Lock Settings", repr=False
    )
    color: ColorDetectSettings = Field(
        default_factory=ColorDetectSettings, description="Color Detection Settings"
    )
    models: List = Field(..., description="Models configuration", exclude=True)

    available_models: List[BaseModelConfig] = Field(
        None, description="Available models"
    )

    def get_lock_settings(self):
        return self.locks

    @model_validator(mode="after")
    def _validate_model_after(self) -> "Settings":
        v = []
        models = self.models
        if models:
            disabled_models: list = []
            for model in models:
                if not model:
                    logger.warning(
                        f"Model is empty, this usually means there is a formatting error "
                        f"(or an empty model; just a '-') in your config file! Skipping"
                    )
                    continue
                final_model_config = None
                if model.get("enabled", True) is True:
                    _model: Union[
                        RekognitionModelConfig,
                        TRTModelConfig,
                        ORTModelConfig,
                        TPUModelConfig,
                        FaceRecognitionLibModelConfig,
                        ALPRModelConfig,
                        CV2YOLOModelConfig,
                        TorchModelConfig,
                        None,
                    ] = None
                    _framework = model.get("framework", ModelFrameWork.OPENCV)
                    _sub_fw = model.get("sub_framework", OpenCVSubFrameWork.DARKNET)
                    _options = model.get("detection_options", {})
                    _type = model.get("model_type", ModelType.OBJECT)
                    logger.debug(
                        f"Settings: validate available models => {_framework = } -- {_sub_fw = } -- {_type = } -- {_options = } filename: {model.get('input', 'No input file defined!')}"
                    )
                    if _framework == ModelFrameWork.HTTP:
                        if _sub_fw == HTTPSubFrameWork.REKOGNITION:
                            model["model_type"] = ModelType.OBJECT
                            model["processor"] = ModelProcessor.NONE
                            final_model_config = RekognitionModelConfig(**model)
                    elif _framework == ModelFrameWork.FACE_RECOGNITION:
                        model["model_type"] = ModelType.FACE
                        model["detection_options"] = (
                            FaceRecognitionLibModelDetectionOptions(**_options)
                        )
                        final_model_config = FaceRecognitionLibModelConfig(**model)

                    elif _framework == ModelFrameWork.ALPR:
                        model["model_type"] = ModelType.ALPR
                        api_type = model.get("api_type", "local")
                        config = ALPRModelConfig(**model)
                        if _sub_fw == ALPRSubFrameWork.OPENALPR:
                            if api_type == ALPRAPIType.LOCAL:
                                config.detection_options = OpenALPRLocalModelOptions(
                                    **_options
                                )
                            elif api_type == ALPRAPIType.CLOUD:
                                config.processor = ModelProcessor.NONE
                                config.detection_options = OpenALPRCloudModelOptions(
                                    **_options
                                )
                        elif _sub_fw == ALPRSubFrameWork.PLATE_RECOGNIZER:
                            if api_type == ALPRAPIType.CLOUD:
                                config.processor = ModelProcessor.NONE
                                config.detection_options = PlateRecognizerModelOptions(
                                    **_options
                                )
                            elif api_type == ALPRAPIType.LOCAL:
                                raise NotImplementedError(
                                    "Plate Recognizer Local API not implemented"
                                )

                        final_model_config = config

                    elif _framework == ModelFrameWork.ORT:
                        final_model_config = ORTModelConfig(**model)
                        final_model_config.detection_options = ORTModelOptions(
                            **_options
                        )

                    elif _framework == ModelFrameWork.OPENCV:
                        config = None
                        not_impl = [
                            OpenCVSubFrameWork.TENSORFLOW,
                            OpenCVSubFrameWork.TORCH,
                            OpenCVSubFrameWork.CAFFE,
                            OpenCVSubFrameWork.VINO,
                        ]
                        if (
                            _sub_fw == OpenCVSubFrameWork.DARKNET
                            or _sub_fw == OpenCVSubFrameWork.ONNX
                        ):
                            config = CV2YOLOModelConfig(**model)
                            config.detection_options = CV2YOLOModelOptions(**_options)

                        elif _sub_fw in not_impl:
                            raise NotImplementedError(f"{_sub_fw} not implemented")
                        else:
                            raise ValueError(f"Unknown OpenCV sub-framework: {_sub_fw}")
                        if config:
                            final_model_config = config
                    elif _framework == ModelFrameWork.CORAL:
                        config = TPUModelConfig(**model)
                        config.detection_options = TPUModelOptions(**_options)
                        final_model_config = config
                    elif _framework == ModelFrameWork.TRT:
                        config = TRTModelConfig(**model)
                        config.detection_options = TRTModelOptions(**_options)
                        final_model_config = config
                    elif _framework == ModelFrameWork.TORCH:
                        config = TorchModelConfig(**model)
                        config.detection_options = TorchModelOptions(**_options)
                        final_model_config = config
                    elif _framework == ModelFrameWork.NETINT:
                        config = NETINTModelConfig(**model)
                        config.detection_options = NETINTModelOptions(**_options)
                        final_model_config = config
                    else:
                        raise NotImplementedError(
                            f"Framework {_framework} not implemented"
                        )

                    # load model here
                    if final_model_config:
                        logger.debug(
                            f"Settings: Adding to available models: {final_model_config.name}"
                        )
                        v.append(final_model_config)
                else:
                    disabled_models.append(model.get("name"))
            if disabled_models:
                logger.debug(f"Skipped these disabled models: {disabled_models}")
        if v:
            self.available_models = v
        return self


def parse_client_config_file(cfg_file: Path) -> Optional[Settings]:
    """Parse the YAML configuration file."""

    cfg: Dict = {}
    _start = time.time()
    raw_config = cfg_file.read_text()

    try:
        cfg = yaml.safe_load(raw_config)
    except yaml.YAMLError:
        logger.error(f"Error parsing the YAML configuration file!")
        raise
    except PermissionError:
        logger.error(f"Error reading the YAML configuration file!")
        raise

    substitutions = cfg.get("substitutions", {})
    testing = cfg.get("testing", {})
    testing = Testing(**testing)
    if testing.enabled:
        logger.info(f"|----- TESTING IS ENABLED! -----|")
        if testing.substitutions:
            logger.info(f"Overriding config:substitutions WITH testing:substitutions")
            substitutions = testing.substitutions

    logger.debug(f"Replacing ${{VARS}} in config:substitutions")
    substitutions = _replace_vars(search_str=str(substitutions), var_pool=substitutions)
    # logger.debug(f"AFTER FIRST SUBSTITUTION: {substitutions = }")
    substitutions = _replace_vars(search_str=str(substitutions), var_pool=substitutions)
    # logger.debug(f"AFTER SECOND SUBSTITUTION: {substitutions = }")
    if inc_file := substitutions.get("IncludeFile"):
        inc_file = Path(inc_file)
        logger.debug(f"PARSING IncludeFile: {inc_file.as_posix()}")
        if inc_file.is_file():
            inc_vars = yaml.safe_load(inc_file.read_text())
            if "server" in inc_vars:
                inc_vars = inc_vars["server"]
                # logger.debug(
                #     f"Loaded {len(inc_vars)} substitution from IncludeFile {inc_file} => {inc_vars}"
                # )
                # check for duplicates
                for k in inc_vars:
                    if k in substitutions:
                        logger.warning(
                            f"Duplicate substitution variable '{k}' in IncludeFile {inc_file} - "
                            f"IncludeFile overrides config file"
                        )

                substitutions.update(inc_vars)
            else:
                logger.warning(
                    f"IncludeFile [{inc_file}] does not have a 'client' section - skipping"
                )
        else:
            logger.warning(f"IncludeFile {inc_file} is not a file!")
    logger.debug(f"Replacing ${{VARS}} in config")
    cfg = _replace_vars(raw_config, substitutions)
    logger.debug(f"perf: Config file loaded in {time.time() - _start:.5f} seconds")

    _cfg = Settings(**cfg)

    # logger.debug(f"cfg file parsed into a class = {_cfg}")
    return _cfg


class APIDetector:
    """ML detector API class.
    Specify a processor type and then load the model into processor memory. run an inference on the processor
    """

    lp: str = "APIDetector:"

    id: uuid.UUID = Field(None, description="Model ID")
    _config: Union[
        BaseModelConfig,
        CV2YOLOModelConfig,
        ALPRModelConfig,
        FaceRecognitionLibModelConfig,
        DeepFaceModelConfig,
        TPUModelConfig,
        TorchModelConfig,
        "ORTModelConfig",
        "TRTModelConfig",
    ]
    _options: Union[
        BaseModelOptions,
        OpenALPRLocalModelOptions,
        OpenALPRCloudModelOptions,
        PlateRecognizerModelOptions,
        ALPRModelOptions,
        FaceRecognitionLibModelDetectionOptions,
        CV2YOLOModelOptions,
        TorchModelOptions,
        "ORTModelOptions",
        "TRTModelOptions",
        "NETINTModelOptions"
    ]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.config})"

    def __get__(self, instance, owner):
        return self

    def __init__(
        self,
        model_config: Union[
            BaseModelConfig,
            CV2YOLOModelConfig,
            ALPRModelConfig,
            FaceRecognitionLibModelConfig,
            DeepFaceModelConfig,
            TPUModelConfig,
            TorchModelConfig,
            "ORTModelConfig",
            "TRTModelConfig",
        ],
    ):
        from ..ML.Detectors.opencv.cv_darknet import CV2DarkNetDetector
        from ..ML.Detectors.coral_edgetpu import TpuDetector
        from ..ML.Detectors.face_recognition import FaceRecognitionLibDetector
        from ..ML.Detectors.alpr import PlateRecognizer, OpenAlprCmdLine, OpenAlprCloud
        from ..ML.Detectors.aws_rekognition import AWSRekognition
        from ..ML.Detectors.torch.torch_base import TorchDetector
        from ..ML.Detectors.onnx_runtime import ORTDetector

        self.config = model_config
        self.id = self.config.id
        self.options = model_config.detection_options
        self.model: Optional[
            Union[
                TorchDetector,
                TpuDetector,
                CV2DarkNetDetector,
                FaceRecognitionLibDetector,
                ORTDetector,
                "TensorRtDetector",
                PlateRecognizer,
                OpenAlprCmdLine,
                OpenAlprCloud,
                AWSRekognition,
            ]
        ] = None
        self._load_model()

    def warm_up(self):
        """
        After loading a model, synchronously run an inference to warm up the model. Its synchronous to avoid OOM errors
        when multiple models are loaded at the same time. This is a blocking call.
        """
        lp: str = "%swarm_up:" % self.lp
        if self.model:
            self.model.warm_up()
        else:
            logger.warning(f"{lp} Model not loaded, cannot warm up!")

    @property
    def options(
        self,
    ) -> Union[
        BaseModelOptions,
        OpenALPRLocalModelOptions,
        OpenALPRCloudModelOptions,
        PlateRecognizerModelOptions,
        ALPRModelOptions,
        FaceRecognitionLibModelDetectionOptions,
        CV2YOLOModelOptions,
        TorchModelOptions,
        "ORTModelOptions",
        "TRTModelOptions",
    ]:
        return self._options

    @options.setter
    def options(
        self,
        options: Union[
            BaseModelOptions,
            OpenALPRLocalModelOptions,
            OpenALPRCloudModelOptions,
            PlateRecognizerModelOptions,
            ALPRModelOptions,
            FaceRecognitionLibModelDetectionOptions,
            CV2YOLOModelOptions,
            TorchModelOptions,
            "ORTModelOptions",
            "TRTModelOptions",
        ],
    ):
        self._options = options

    def _load_model(
        self,
        config: Optional[
            Union[
                BaseModelConfig,
                CV2YOLOModelConfig,
                ALPRModelConfig,
                FaceRecognitionLibModelConfig,
                DeepFaceModelConfig,
                TPUModelConfig,
                TorchModelConfig,
                ORTModelConfig,
                TRTModelConfig,
                RekognitionModelConfig,
            ]
        ] = None,
    ):
        """Load the model, warming them up is separate"""

        from ..ML.Detectors.face_recognition import FaceRecognitionLibDetector
        from ..ML.Detectors.alpr import PlateRecognizer, OpenAlprCmdLine, OpenAlprCloud
        from ..ML.Detectors.aws_rekognition import AWSRekognition

        self.model = None
        _proc_available: Optional[bool] = False
        if config:
            self.config = config
        if not self.config.processor:
            self.config.processor = ModelProcessor.CPU

        _proc_available = self.is_processor_available()
        if _proc_available is False:
            _failed_proc = self.config.processor.value

            if self.config.framework == ModelFrameWork.CORAL:
                self.config.processor = ModelProcessor.TPU
            elif self.config.framework == ModelFrameWork.HTTP:
                self.config.processor = ModelProcessor.NONE
            elif self.config.framework == ModelFrameWork.NETINT:
                self.config.processor = ModelProcessor.QUADRA
            else:
                self.config.processor = ModelProcessor.CPU
            logger.warning(
                f"{_failed_proc} is not available to the {self.config.framework} framework! "
                f"Switching to {self.config.processor}"
            )

        elif _proc_available is None:
            from ..app import get_global_config

            for _model in get_global_config().available_models:
                if _model.id == self.id:
                    get_global_config().available_models.remove(_model)
                    break
            raise ImportError(
                f"Library missing, cannot create detector for {self.config.name}"
            )

        try:
            if self.config.framework == ModelFrameWork.OPENCV:
                if self.config.sub_framework == OpenCVSubFrameWork.DARKNET:
                    from ..ML.Detectors.opencv.cv_darknet import CV2DarkNetDetector

                    self.model = CV2DarkNetDetector(self.config)

                elif self.config.sub_framework == OpenCVSubFrameWork.ONNX:
                    from ..ML.Detectors.opencv.onnx import CV2ONNXDetector

                    self.model = CV2ONNXDetector(self.config)
            elif self.config.framework == ModelFrameWork.TRT:
                from ..ML.Detectors.tensorrt.trt_base import TensorRtDetector

                self.model = TensorRtDetector(self.config)
            elif self.config.framework == ModelFrameWork.ORT:
                from ..ML.Detectors.onnx_runtime import ORTDetector

                self.model = ORTDetector(self.config)
            elif self.config.framework == ModelFrameWork.HTTP:
                sub_fw = self.config.sub_framework
                if sub_fw == HTTPSubFrameWork.REKOGNITION:
                    self.model = AWSRekognition(self.config)
                elif sub_fw == HTTPSubFrameWork.NONE:
                    raise RuntimeError(
                        f"Invalid HTTP sub framework {sub_fw}, YOU MUST CHOOSE A sub_framework!"
                    )
                else:
                    raise RuntimeError(f"Invalid HTTP sub framework: {sub_fw}")
            elif self.config.framework == ModelFrameWork.FACE_RECOGNITION:
                self.model = FaceRecognitionLibDetector(self.config)
            elif self.config.framework == ModelFrameWork.ALPR:
                if self.config.service == ALPRService.PLATE_RECOGNIZER:
                    self.model = PlateRecognizer(self.config)
                elif self.config.service == ALPRService.OPENALPR:
                    if self.config.api_type == ALPRAPIType.LOCAL:
                        self.model = OpenAlprCmdLine(self.config)
                    elif self.config.api_type == ALPRAPIType.CLOUD:
                        self.model = OpenAlprCloud(self.config)

            elif self.config.framework == ModelFrameWork.CORAL:
                from ..ML.Detectors.coral_edgetpu import TpuDetector

                self.model = TpuDetector(self.config)

            elif self.config.framework == ModelFrameWork.TORCH:
                from ..ML.Detectors.torch.torch_base import TorchDetector

                self.model = TorchDetector(self.config)

            elif self.config.framework == ModelFrameWork.NETINT:
                from ..ML.Detectors.netint import NETINTDetector
                self.model = NETINTDetector(self.config)

            else:
                logger.warning(
                    f"CANT CREATE DETECTOR -> Framework NOT IMPLEMENTED!!! {self.config.framework}"
                )
        except Exception as e:
            logger.warning(
                f"Error loading model ({self.config.name}): {e}", exc_info=True
            )
            raise e

    def is_processor_available(self) -> bool:
        """Check if the processor is available"""
        available = False
        processor = self.config.processor
        framework = self.config.framework
        logger.debug(
            f"Checking if {processor} is available to use for {framework} ({self.config.name})"
        )
        if processor == ModelProcessor.NONE:
            available = True
        if framework == ModelFrameWork.ALPR or framework == ModelFrameWork.HTTP:
            available = True
        elif not processor:
            available = True
        elif processor == ModelProcessor.CPU:
            if framework == ModelFrameWork.CORAL:
                logger.error(f"{processor} is not supported for {framework}")
            else:
                available = True

        elif processor == ModelProcessor.TPU:
            if framework == ModelFrameWork.CORAL:
                try:
                    from pycoral.utils.edgetpu import list_edge_tpus
                except ImportError:
                    logger.warning(
                        "pycoral not installed, cannot load any models that use the TPU processor"
                    )
                    available = None
                else:
                    tpus = list_edge_tpus()
                    logger.debug(f"TPU devices found: {tpus}")
                    if tpus:
                        available = True
                    else:
                        logger.warning(
                            "No TPU devices found, cannot load any models that use the TPU processor"
                        )
            else:
                logger.warning("TPU processor is only available for Coral models!")
        elif processor == ModelProcessor.GPU:
            if framework == ModelFrameWork.OPENCV:
                try:
                    import cv2.cuda
                except ImportError:
                    logger.warning(
                        "OpenCV does not have CUDA enabled/compiled, cannot load any models that use the GPU processor"
                    )
                    available = None
                else:
                    # wrap in try block as this will throw an exception if no CUDA devices are found
                    try:
                        if not (cuda_devices := cv2.cuda.getCudaEnabledDeviceCount()):
                            logger.warning(
                                f"No CUDA devices found for '{self.config.name}'"
                            )
                    except Exception as cv2_cuda_exception:
                        logger.warning(
                            f"'{self.config.name}' Error getting CUDA device count: {cv2_cuda_exception}"
                        )
                        available = None
                    else:
                        logger.debug(
                            f"Found {cuda_devices} CUDA device(s) that OpenCV can use for '{self.config.name}'"
                        )
                        available = True
            elif framework == ModelFrameWork.ORT:
                try:
                    import onnxruntime as ort
                except ImportError:
                    logger.warning(
                        "onnxruntime not installed, cannot load any models that use it"
                    )
                    available = None
                else:
                    if ort.get_device() == "GPU":
                        available = True
                    else:
                        logger.warning(f"No GPU devices found for '{self.config.name}'")
            elif framework in (ModelFrameWork.TORCH):
                try:
                    import torch
                except ImportError:
                    logger.warning(
                        "torch not installed, cannot load any models that use torch"
                    )
                    available = None
                    torch = None
                else:
                    if not torch.cuda.is_available():
                        logger.warning(
                            "No CUDA devices found, cannot load any models that use the GPU processor"
                        )
                    else:
                        available = True
            elif framework == ModelFrameWork.FACE_RECOGNITION:
                try:
                    import dlib
                except ImportError:
                    logger.warning(
                        "dlib not installed, cannot load any models that use dlib!"
                    )
                    available = None
                else:
                    try:
                        if dlib.DLIB_USE_CUDA and dlib.cuda.get_num_devices() >= 1:
                            available = True
                    except Exception as dlib_cuda_exception:
                        logger.warning(
                            f"'{self.config.name}' Error getting dlib CUDA device count: {dlib_cuda_exception}"
                        )
                        available = False

            elif framework == ModelFrameWork.DEEPFACE:
                logger.warning("WORKING ON DeepFace models!")
                available = False
            elif framework == ModelFrameWork.TRT:
                available = True
        elif processor == ModelProcessor.QUADRA:
            if framework == ModelFrameWork.NETINT:
                available = True
        logger.debug(
            f"{processor} is {'NOT ' if not available else ''}available for {framework} - '{self.config.name}'"
        )
        return available

    async def detect(self, images: List[np.ndarray]) -> DetectionResults:
        """Detect objects in the images

        :param images: List of images to detect objects in
        :return: `DetectionResult` object
        """
        if self.model is None:
            logger.warning(
                f"Detector for {self.config.name} is not loaded, cannot detect objects!"
            )
            return
        return await self.model.detect(images)


class GlobalConfig(BaseModel, arbitrary_types_allowed=True):

    available_models: List[
        Union[
            BaseModelConfig,
            CV2YOLOModelConfig,
            ALPRModelConfig,
            FaceRecognitionLibModelConfig,
            TorchModelConfig,
            TRTModelConfig,
            ORTModelConfig,
            TPUModelConfig,
            RekognitionModelConfig,
        ]
    ] = Field(default_factory=list, description="Available models, call by ID")
    config: Settings = Field(
        default=None, description="Global config, loaded from YAML config file"
    )
    detectors: List[APIDetector] = Field(
        default_factory=list, description="Loaded Detectors"
    )
    user_db: None = Field(None, description="User Database (TinyDB)")
    async_locks: Dict[ModelProcessor, BoundedSemaphore] = Field(default_factory=dict)
    color_detector: Optional[ColorDetector] = None

    def create_detector(self, model: BaseModelConfig) -> Optional[APIDetector]:
        logger.debug(f"Attempting to create new detector for '{model.name}'")
        _s = time.time()
        ret_ = None
        try:
            ret_ = APIDetector(model)
        except ImportError as e:
            logger.warning(e, exc_info=True)
            _ret = None
        except Exception as e:
            logger.error(e, exc_info=True)
            _ret = None
        else:
            logger.debug(
                f"Detector for '{model.name}' created in {time.time() - _s:.5f} seconds"
            )
            # warm the model up
            # ret_.warm_up()
            self.detectors.append(ret_)

        if not ret_:
            logger.error(
                f"Unable to create detector for '{model.name}', removing from available models..."
            )
            if model in self.available_models:
                self.available_models.remove(model)

        return ret_

    def get_detector(self, model: BaseModelConfig) -> Optional[APIDetector]:
        """Get a detector by ID (UUID4)"""
        ret_: Optional[APIDetector] = None
        for detector in self.detectors:
            if detector.config.id == model.id:
                ret_ = detector
                break
        if not ret_:
            ret_ = self.create_detector(model)
        return ret_


class ServerEnvVars(BaseSettings):
    """Server Environment Variables using pydantic v2 BaseSettings

     NOTE: the name of the attribute must match the name of the environment variable,
    you can set an env-prefix in the model config

    """

    model_config = SettingsConfigDict(
        env_prefix="ML_SERVER_", case_sensitive=True, extra="allow"
    )

    # With the env_prefix set, the env var name will be: ML_SERVER_CONF_FILE
    conf_file: str = Field(None, description="Server YAML config file")
