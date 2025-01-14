#!/usr/bin/env python3

import argparse
import functools
import grp
import logging
import os
import pwd
import random
import re
import shutil
import string
import subprocess
import sys
import tempfile
import time
import venv
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from typing import Optional, Tuple, Union, List, Dict, Pattern

_simple_re: Pattern = re.compile(r"(?<!\\)\$([A-Za-z0-9_]+)")
_extended_re: Pattern = re.compile(r"(?<!\\)\$\{([A-Za-z0-9_]+)((:?-)([^}]+))?}")

VENV_NAME: str = "ZoMi_Server"

# Change these if you want to install to a different location
DEFAULT_BASE_DIR = "/opt/zomi/server"
DEFAULT_DATA_DIR = f"{DEFAULT_BASE_DIR}/data"
DEFAULT_CONFIG_DIR = f"{DEFAULT_BASE_DIR}/conf"
DEFAULT_LOG_DIR = f"{DEFAULT_BASE_DIR}/logs"
DEFAULT_TMP_DIR = f"{tempfile.gettempdir()}/zomi/server"
DEFAULT_MODEL_DIR = f"{DEFAULT_BASE_DIR}/models"
DEFAULT_LOG_FILE = "./zomi-server_install.log"
DEFAULT_SYSTEM_CREATE_PERMISSIONS = 0o755
# config files will have their permissions adjusted to this
DEFAULT_CONFIG_CREATE_PERMISSIONS = 0o755
# default ML models to install (SEE: available_models{})
DEFAULT_MODELS = [
    "yolov8n",
    "yolo_nas_s",
]
REPO_BASE = Path(__file__).parent.parent
EXAMPLES_DIR = Path(__file__).parent
_ENV: Dict[str, str] = {}
THREADS: Dict[str, Thread] = {}

# Do not change these unless you know what you are doing
available_models = {
    "yolov8n": {
        "folder": "yolo",
        "model": [
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        ],
        "config": [],
    },
    "yolov8s": {
        "folder": "yolo",
        "model": [
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
        ],
        "config": [],
    },
    "yolov8m": {
        "folder": "yolo",
        "model": [
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"
        ],
        "config": [],
    },
    "yolov8l": {
        "folder": "yolo",
        "model": [
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt"
        ],
        "config": [],
    },
    "yolov8x": {
        "folder": "yolo",
        "model": [
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"
        ],
        "config": [],
    },
    "yolo_nas_s": {
        "folder": "yolo",
        "model": [
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo_nas_s.pt"
        ],
        "config": [],
    },
    "yolo_nas_m": {
        "folder": "yolo",
        "model": [
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo_nas_m.pt"
        ],
        "config": [],
    },
    "yolo_nas_l": {
        "folder": "yolo",
        "model": [
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo_nas_l.pt"
        ],
        "config": [],
    },
    "yolov4": {
        "folder": "yolo",
        "model": [
            # "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights",
            "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4_new.weights",
        ],
        "config": [
            # "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4_new.cfg",
        ],
    },
    "yolov4_tiny": {
        "folder": "yolo",
        "model": [
            "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"
        ],
        "config": [
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
        ],
    },
    "yolov4_p6": {
        "folder": "yolo",
        "model": [
            "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-p6.weights"
        ],
        "config": [
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-p6.cfg"
        ],
    },
    "yolov7": {
        "folder": "yolo",
        "model": [
            "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov7.weights"
        ],
        "config": [
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov7.cfg"
        ],
    },
    "yolov7_tiny": {
        "folder": "yolo",
        "model": [
            "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov7-tiny.weights"
        ],
        "config": [
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov7-tiny.cfg"
        ],
    },
    "yolov7x": {
        "folder": "yolo",
        "model": [
            "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov7x.weights"
        ],
        "config": [
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov7x.cfg"
        ],
    },
    "coral_tpu": {
        "folder": "coral_tpu",
        "model": [
            "https://github.com/google-coral/edgetpu/raw/master/test_data/"
            "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
            "https://github.com/google-coral/test_data/raw/master/"
            "ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite",
            "https://github.com/google-coral/test_data/raw/master/"
            "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite",
            "https://github.com/google-coral/test_data/raw/master/"
            "tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite",
            "https://github.com/google-coral/test_data/raw/master/"
            "efficientdet_lite3_512_ptq_edgetpu.tflite",
        ],
        "config": [],
    },
}
# Logging
logger = logging.getLogger("zomiAPI-inst")
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter(
    "%(asctime)s.%(msecs)d %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d -> %(message)s",
    "%m/%d/%y %H:%M:%S",
)
console = logging.StreamHandler(stream=sys.stdout)
console.setFormatter(log_formatter)
logger.addHandler(console)

# Misc.
__version__ = "0.0.1a5"
__dependencies__ = "psutil", "requests", "tqdm", "distro"
__doc__ = """Install ZoMi ML Server"""

# Logic
tst_msg_wrap = "[testing!!]", "[will not actually execute]"


# parse .env file using pyenv
def parse_env_file(env_file: Path) -> None:
    """Parse .env file using python-dotenv.

    :param env_file: Path to .env file.
    :type env_file: Path
    :returns: Dict of parsed .env file.
    """
    try:
        import dotenv
    except ImportError:
        logger.warning(f"python-dotenv not installed, skipping {env_file}")
    else:
        global _ENV
        dotenv_vals = dotenv.dotenv_values(env_file)
        _ENV.update(dotenv_vals)


def test_msg(msg: str, level: Optional[Union[str, int]] = None):
    """Print test message. Changes stack level to 2 to show caller of test_msg."""
    if testing:
        logger.warning(f"{tst_msg_wrap[0]} {msg} {tst_msg_wrap[1]}", stacklevel=2)
    else:
        if level in ("debug", logging.DEBUG, None):
            logger.debug(msg, stacklevel=2)
        elif level in ("info", logging.INFO):
            logger.info(msg, stacklevel=2)
        elif level in ("warning", logging.WARNING):
            logger.warning(msg, stacklevel=2)
        elif level in ("error", logging.ERROR):
            logger.error(msg, stacklevel=2)
        elif level in ("critical", logging.CRITICAL):
            logger.critical(msg, stacklevel=2)
        else:
            logger.info(msg, stacklevel=2)


def check_imports():
    import importlib

    ret = True
    for imp_name in __dependencies__:
        try:
            importlib.import_module(imp_name)
        except ImportError:
            _msg = (
                f"Missing python module dependency: {imp_name}"
                f":: Please install the python package"
            )
            logger.error(_msg)
            print(_msg)
            ret = False
        else:
            logger.debug(f"Found python module dependency: {imp_name}")
    return ret


def get_web_user() -> Tuple[Optional[str], Optional[str]]:
    """Get the user that runs the web server using psutil library.

    :returns: The user that runs the web server.
    :rtype: str

    """
    try:
        import psutil
    except ImportError:
        logger.warning("psutil not installed, cannot get web server user")
        return None, None
    # get group name from gid

    www_daemon_names = ("httpd", "hiawatha", "apache", "nginx", "lighttpd", "apache2")
    hits = []
    proc_names = []
    for proc in psutil.process_iter():
        if any(x.startswith(proc.name()) for x in www_daemon_names):
            uname, ugroup = proc.username(), grp.getgrgid(proc.gids().real).gr_name
            logger.debug(f"Found web server process: {proc.name()} ({uname}:{ugroup})")
            hits.append((uname, ugroup))
            proc_names.append(proc.name())
    proc_names = list(set(proc_names))
    if len(hits) > 1:
        # sort by uid high to low
        hits = sorted(hits, key=lambda x: pwd.getpwnam(x[0]).pw_uid, reverse=True)
        logger.warning(
            f"Multiple web server processes found ({proc_names}) - The list is sorted to account "
            f"for root dropping privileges. Hopefully the first entry is the correct one -> {hits}"
        )

    if hits:
        return hits[0]
    return None, None


def get_models():
    global models, no_models, THREADS
    # check models
    if no_models:
        logger.info(
            " --no-models requested when the server is being installed?"
            " Skipping model download..."
        )
        return
    if not models:
        logger.info(f"No models specified, using default models {DEFAULT_MODELS}")
        if interactive:
            x = input(f"Download default models? [Y/n]... ")
            if x.casefold() == "n":
                logger.info("Skipping model download...")
                models = []
            else:
                models = DEFAULT_MODELS
        else:
            logger.info("Skipping download of default models...")

    if models:
        for model in models:
            model = model.strip().casefold()
            if model not in available_models.keys():
                logger.error(
                    f"Invalid model '{model}' -  Allowed models: {', '.join(available_models.keys())}"
                )
            else:
                logger.info(f"Downloading model data: {model}")
                model_data = available_models[model]
                model_folder = model_dir / model_data["folder"]
                _model = model_data["model"]
                _config = model_data["config"]

                if _model:
                    for model_url in model_data["model"]:
                        model_file = model_folder / Path(model_url).name
                        if model_file.exists():
                            if not force_models:
                                logger.warning(
                                    f"Model file '{model_file}' already exists, skipping..."
                                )
                                continue
                            else:
                                logger.info(
                                    f"--force-model passed via CLI - Model file "
                                    f"'{model_file}' already exists, overwriting..."
                                )
                        else:
                            logger.info(
                                f"Model file ({model_file}) does not exist, downloading..."
                            )
                        # Thread it
                        THREADS[model_file.name] = Thread(
                            target=download_file,
                            args=(
                                model_url,
                                model_file,
                                ml_user,
                                ml_group,
                                cfg_create_mode,
                            ),
                        )
                        THREADS[model_file.name].start()
                if _config:
                    for config_url in model_data["config"]:
                        config_file = model_folder / Path(config_url).name
                        if config_file.exists():
                            if not force_models:
                                logger.warning(
                                    f"Config file '{config_file}' already exists, skipping..."
                                )
                                continue
                            else:
                                logger.info(
                                    f"--force-model passed via CLI - Config file "
                                    f"'{config_file}' already exists, overwriting..."
                                )
                        THREADS[config_file.name] = Thread(
                            target=download_file,
                            args=(
                                config_url,
                                config_file,
                                ml_user,
                                ml_group,
                                cfg_create_mode,
                            ),
                        )
                        THREADS[config_file.name].start()


def parse_cli():
    global args, models

    parser = argparse.ArgumentParser(description=__doc__)
    processor_group = parser.add_mutually_exclusive_group(required=True)
    processor_group.add_argument(
        "--cpu",
        action="store_true",
        dest="cpu",
        help="Install CPU only packages for pytorch / onnx-runtime",
    )
    processor_group.add_argument(
        "--gpu",
        dest="gpu",
        help="Install GPU only packages for pytorch / onnx-runtime",
        choices=("cuda12.1", "cuda11.8", "rocm6.0"),
        default=None,
        nargs=1,
    )
    processor_group.add_argument(
        "--tpu",
        action="store_true",
        dest="tpu",
        help="Install TPU only packages for pytorch / onnx-runtime",
    )
    parser.add_argument("--venv-only", action="store_true", help="Install venv only", dest="venv_only")
    parser.add_argument("-F", "--force", action="store_true", help="Force installation (overwrite existing files)", dest="force")
    parser.add_argument(
        "--opencv",
        action="store_true",
        dest="opencv",
        help="Install OpenCV (Without CUDA)",
    )
    parser.add_argument(
        "--face-recognition",
        "--ageitgey",
        action="store_true",
        dest="face_rec",
        help="Install ageitgey (Dlib based) face detection/recognition framework (legacy)",
    )

    parser.add_argument(
        "--no-cache-dir",
        action="store_true",
        dest="no_cache",
        help="Do not use cache directory for pip install",
    )
    parser.add_argument(
        "--editable",
        action="store_true",
        dest="pip_install_editable",
        help="Use the --editable flag when installing via pip (a git pull will update "
        "the installed package) BE AWARE: you must keep the git source directory intact for this to work",
    )

    parser.add_argument(
        "--route-host",
        dest="route_host",
        default=None,
        type=str,
        help="Server address [Optional: 0.0.0.0]",
    )
    parser.add_argument(
        "--route-port",
        dest="route_port",
        default=None,
        type=str,
        help="Server port [Optional: 5000]",
    )
    parser.add_argument("--route-key", dest="route_key", default=None, type=str)

    parser.add_argument(
        "--config-only",
        dest="config_only",
        action="store_true",
        help="Install config files only, used in conjunction with --secret-only",
    )

    parser.add_argument(
        "--secrets-only",
        dest="secrets_only",
        action="store_true",
        help="Install secrets file only, used in conjunction with --config-only",
    )

    parser.add_argument(
        "--interactive",
        "-I",
        action="store_true",
        dest="interactive",
        help="Run in interactive mode (WIP)",
    )
    parser.add_argument(
        "--models-only",
        "--only-models",
        dest="models_only",
        action="store_true",
        help="Install models only",
    )
    parser.add_argument(
        "--add-model",
        action="append",
        dest="models",
        help=f"Download model files (can be used several time --add-model yolov4 "
        f"--add-model yolov7_tiny) Default: {' '.join(DEFAULT_MODELS)}",
        default=DEFAULT_MODELS,
        choices=available_models.keys(),
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        dest="all_models",
        help="Download all available model files",
    )
    parser.add_argument(
        "--text-models",
        type=str,
        dest="text_models",
        help="Download model files designated by a comma delimited string of model names [yolov4,yolov7,etc]",
    )

    parser.add_argument(
        "--force-models",
        action="store_true",
        dest="force_models",
        help="Force model installation [Overwrite existing model files]",
    )
    parser.add_argument(
        "--no-models",
        dest="no_models",
        action="store_true",
        help="Do not install models",
    )

    parser.add_argument(
        "--install-log",
        type=str,
        default=DEFAULT_LOG_FILE,
        help="Log file to write installation logs to",
    )
    parser.add_argument(
        "--env-file", type=Path, help="Path to .env file (requires python-dotenv)"
    )
    parser.add_argument(
        "--dir-config",
        help=f"Directory where config files are held [Default: {DEFAULT_CONFIG_DIR}]",
        default=DEFAULT_CONFIG_DIR,
        type=Path,
        dest="config_dir",
    )
    parser.add_argument(
        "--dir-data",
        help=f"Directory where variable data is held [Default: {DEFAULT_DATA_DIR}]",
        dest="data_dir",
        default=DEFAULT_DATA_DIR,
        type=Path,
    )
    parser.add_argument(
        "--dir-model",
        type=str,
        help=f"ML model base directory [Default: {DEFAULT_MODEL_DIR}]",
        default="",
        dest="model_dir",
    )
    parser.add_argument(
        "--dir-temp",
        "--dir-tmp",
        type=Path,
        help=f"Temp files directory [Default: {DEFAULT_TMP_DIR}]",
        default=DEFAULT_TMP_DIR,
        dest="tmp_dir",
    )
    parser.add_argument(
        "--dir-log",
        help=f"Directory where logs will be stored [Default: {DEFAULT_LOG_DIR}]",
        default=DEFAULT_LOG_DIR,
        dest="log_dir",
        type=Path,
    )

    parser.add_argument(
        "--user",
        "-U",
        help="User to install as "
        "[leave empty to auto-detect what user runs the web server] "
        "(Change if installing server on a remote host)",
        type=str,
        dest="ml_user",
        default="",
    )
    parser.add_argument(
        "--group",
        "-G",
        help="Group member to install as [leave empty to auto-detect what "
        "group member runs the web server] (Change if installing server "
        "on a remote host)",
        type=str,
        dest="ml_group",
        default="",
    )

    parser.add_argument(
        "--debug", "-D", help="Enable debug logging", action="store_true", dest="debug"
    )
    parser.add_argument(
        "--dry-run",
        "--test",
        "-T",
        action="store_true",
        dest="test",
        help="Run in test mode, no actions are actually executed.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    parser.add_argument(
        "--system-create-permissions",
        help=f"ZoMi system octal [0o] file permissions [Default: {oct(DEFAULT_SYSTEM_CREATE_PERMISSIONS)}]",
        type=lambda x: int(x, 8),
        default=DEFAULT_SYSTEM_CREATE_PERMISSIONS,
    )
    parser.add_argument(
        "--config-create-permissions",
        help=f"Config files (server.yml, client.yml, secrets.yml) octal [0o] file permissions "
        f"[Default: {oct(DEFAULT_CONFIG_CREATE_PERMISSIONS)}]",
        type=lambda x: int(x, 8),
        default=DEFAULT_CONFIG_CREATE_PERMISSIONS,
    )

    return parser.parse_args()


def chown(path: Path, user: Union[str, int], group: Union[str, int]):
    if isinstance(user, str):
        user = pwd.getpwnam(user).pw_uid
    if isinstance(group, str):
        group = grp.getgrnam(group).gr_gid
    test_msg(f"chown {user}:{group} {path}")
    if not testing:
        os.chown(path, user, group)


def chmod(path: Path, mode: int):
    test_msg(f"chmod OCTAL: {mode:o} RAW: {mode} => {path}")
    if not testing:
        os.chmod(path, mode)


def chown_mod(path: Path, user: Union[str, int], group: Union[str, int], mode: int):
    chown(path, user, group)
    chmod(path, mode)


def create_dir(path: Path, user: Union[str, int], group: Union[str, int], mode: int):
    msg = f"Created directory: {path} with user:group:permission [{user}:{group}:{mode:o}]"
    test_msg(msg)
    if not testing:
        path.mkdir(parents=True, exist_ok=True, mode=mode)
        chown_mod(path, user, group, mode)


def show_config(cli_args: argparse.Namespace):
    logger.info(f"Configuration:")
    for key, value in vars(cli_args).items():
        if key.endswith("_permissions"):
            value = oct(value)
        elif key == "models":
            value = ", ".join(value)
        elif isinstance(value, Path):
            value = value.expanduser().resolve().as_posix()
        logger.info(f"    {key}: {value.__repr__()}")
    if args.interactive:
        input(
            "Press Enter to continue if all looks fine, otherwise use 'Ctrl+C' to exit and edit CLI options..."
        )
    else:
        msg = f"This is a non-interactive{' TEST' if testing else ''} session, continuing with installation... "
        if testing:
            logger.info(msg)
        else:
            logger.info(f"{msg} in 5 seconds, press 'Ctrl+C' to exit")
            time.sleep(5)


def do_web_user():
    global ml_user, ml_group
    _group = None
    if not ml_user or not ml_group:
        if not ml_user:
            if interactive:
                ml_user = input("Web server username [Leave blank to try auto-find]: ")
                if not ml_user:
                    ml_user, _group = get_web_user()
                    logger.info(f"Web server user auto-find: {ml_user}")
            else:
                ml_user, _group = get_web_user()
                logger.info(f"Web server user auto-find: {ml_user}")

        if not ml_group:
            if interactive:
                ml_group = input("Web server group [Leave blank to try auto-find]: ")
                if not ml_group:
                    if _group:
                        ml_group = _group
                    else:
                        _, ml_group = get_web_user()
                    logger.info(f"Web server group auto-find: {ml_group}")

            else:
                if _group:
                    ml_group = _group
                else:
                    _, ml_group = get_web_user()
                logger.info(f"Web server group auto-find: {ml_group}")

    if not ml_user or not ml_group:
        _missing = ""
        _u = "user [--user]"
        _g = "group [--group]"
        if not ml_user and ml_group:
            _missing = _u
        elif ml_user and not ml_group:
            _missing = _g
        else:
            _missing = f"{_u} and {_g}"

        msg = f"Unable to determine web server {_missing}, EXITING..."
        if not testing:
            logger.error(msg)
            sys.exit(1)
        else:
            test_msg(msg)


def install_dirs(
    dest_dir: Path,
    default_dir: str,
    dir_type: str,
    sub_dirs: Optional[List[str]] = None,
    perms: int = 0o755,
):
    """Install directories"""
    if sub_dirs is None:
        sub_dirs = []
    if not dest_dir:
        if interactive:
            dest_dir = input(
                f"Set {dir_type} directory [Leave blank to use default {default_dir}]: "
            )

    if not dest_dir:
        if default_dir:
            logger.info(f"Using default {dir_type} directory: {default_dir}")
            dest_dir = default_dir
        else:
            raise ValueError(f"Default {dir_type} directory not set!")

    dest_dir = Path(dest_dir)
    if dir_type.casefold() == "data":
        global data_dir
        data_dir = dest_dir
    elif dir_type.casefold() == "log":
        global log_dir
        log_dir = dest_dir
    elif dir_type.casefold() == "config":
        global cfg_dir
        cfg_dir = dest_dir

    if not dest_dir.exists():
        logger.warning(f"{dir_type} directory {dest_dir} does not exist!")
        if interactive:
            x = input(f"Create {dir_type} directory [Y/n]?... 'n' will exit! ")
            if x.strip().casefold() == "n":
                logger.error(f"{dir_type} directory does not exist, exiting...")
                sys.exit(1)
            create_dir(dest_dir, ml_user, ml_group, perms)
        else:
            test_msg(f"Creating {dir_type} directory...")
            create_dir(dest_dir, ml_user, ml_group, perms)
        # create sub-folders
        if sub_dirs:
            test_msg(
                f"Creating {dir_type} sub-folders: {', '.join(sub_dirs).lstrip(',')}"
            )
            for _sub in sub_dirs:
                _path = dest_dir / _sub
                create_dir(_path, ml_user, ml_group, perms)
    else:
        logger.warning(f"{dir_type} directory {dest_dir} already exists!")


def download_file(url: str, dest: Path, user: str, group: str, mode: int):
    msg = (
        f"Downloading {url}..."
        if not testing
        else f"TESTING if file exists at :: {url}..."
    )
    logger.info(msg)
    import requests
    from tqdm.auto import tqdm

    try:
        r = requests.get(url, stream=True, allow_redirects=True, timeout=5)
        if r:
            file_size = int(r.headers.get("Content-Length", 0))
            dest = dest.expanduser().resolve()
            dest.parent.mkdir(parents=True, exist_ok=True)

            desc = (
                "(Unknown total file size!)"
                if file_size == 0
                else f"Downloading {url} ..."
            )
            r.raw.read = functools.partial(
                r.raw.read, decode_content=True
            )  # Decompress if needed
            with tqdm.wrapattr(
                r.raw, "read", total=file_size, desc=desc, colour="green"
            ) as r_raw:
                do_chown = True
                if testing:
                    do_chown = False
                    logger.info(
                        f"TESTING: File exists at the url for {url.split('/')[-1]}"
                    )
                    # to keep the progress bar, pipe output to /dev/null
                    dest = Path("/dev/null")
                try:
                    with dest.open("wb") as f:
                        shutil.copyfileobj(r_raw, f)
                except Exception as e:
                    logger.error(
                        f"Failed to open or copy data to destination file ({dest}) => {e}"
                    )
                    raise e
                else:
                    logger.info(f"Successfully downloaded {url} to {dest}")
                    if do_chown:
                        chown_mod(dest, user, group, mode)
        else:
            logger.error(f"NO RESPONSE FROM {url} :: Failed to download {url}!")
    except requests.exceptions.ConnectionError:
        logger.error(f"REQUESTS CONNECTION ERROR :: Failed to download {url}!")
        return
    except Exception as e:
        logger.error(f"Failed to download {url}! EXCEPTION :: {e}")


def copy_file(src: Path, dest: Path, user: str, group: str, mode: int):
    msg = f"Copying {src} to {dest}..."
    if not testing:
        if dest.exists():
            if interactive:
                x = input(
                    f"The file {dest} already exists, would you like to remove it and create a new one? [Y/n] "
                )
                if x.casefold() == "n":
                    return
                dest.unlink()
            else:
                if not force:
                    logger.critical(f"File {dest} already exists, this is a fatal error when --force/-F is not supplied. "
                               f"Please remove the existing file and try again.")
                    sys.exit(1)
                else:
                    logger.warning("Overwriting existing file...")
                    dest.unlink()

        # check if the destination directory exists
        if not dest.parent.exists():
            logger.warning(f"Destination directory {dest.parent} does not exist!")
            if interactive:
                x = input(f"Create directory {dest.parent}? [Y/n]... 'n' will exit! ")
                if x.strip().casefold() == "n":
                    logger.error(f"Destination directory does not exist, exiting...")
                    sys.exit(1)
                create_dir(dest.parent, ml_user, ml_group, mode)
            else:
                test_msg(f"Creating destination directory: {dest.parent}")
                create_dir(dest.parent, ml_user, ml_group, mode)
        else:
            chown(dest.parent, ml_user, ml_group)

        logger.info(msg)
        shutil.copy(src, dest)
        chown_mod(dest, user, group, mode)
    else:
        test_msg(msg)


def main():
    install_dirs(
        data_dir,
        DEFAULT_DATA_DIR,
        "Data",
        sub_dirs=[
            "models",
            "push",
            "scripts",
            "images",
            "bin",
            "misc",
        ],
    )
    install_dirs(
        data_dir / "/face_data",
        DEFAULT_DATA_DIR + "/face_data",
        "Face Data",
        sub_dirs=["known", "unknown"],
        perms=0o777,
    )
    install_dirs(
        cfg_dir, DEFAULT_CONFIG_DIR, "Config", sub_dirs=[], perms=cfg_create_mode
    )
    install_dirs(log_dir, DEFAULT_LOG_DIR, "Log", sub_dirs=[], perms=0o777)
    get_models()
    check_backup("secrets")
    do_install()


def check_backup(_inst_type: str) -> None:
    files: List[Optional[Path]] = [x for x in cfg_dir.rglob(f"{_inst_type}.*")]
    if files:
        files = sorted(files, reverse=True)
        _target: Optional[Path] = None
        backup_num = 1
        high = 0
        for _file in files:
            if _file.name.startswith(f"{_inst_type}"):
                if _file.suffix == ".bak":
                    i = int(_file.stem.split(".")[-1])
                    logger.debug(f"Found a {_inst_type} backup numbered: {i}")

                    if i > high:
                        high = backup_num = i + 1
                        logger.debug(
                            f"This backup has a higher number, using it as index: {backup_num = }"
                        )

                elif _file.suffix in (".yml", ".yaml"):
                    logger.debug(f"found existing {_inst_type} config file: {_file}")
                    _target = _file
        if _target:
            file_backup = cfg_dir / f"{_inst_type}.{backup_num}.bak"
            logger.debug(
                f"Backing up existing {_inst_type} config file: {_target} to {file_backup}"
            )
            # Backup existing
            copy_file(_target, file_backup, ml_user, ml_group, cfg_create_mode)
            # install new
            copy_file(
                REPO_BASE / f"configs/example_{_inst_type}.yml",
                cfg_dir / f"{_inst_type}.yml",
                ml_user,
                ml_group,
                cfg_create_mode,
            )
    else:
        logger.debug(f"No existing {_inst_type} config files found!")
        if cfg_dir.is_dir() or testing:
            test_msg(
                f"Creating {_inst_type} config file from example template...", "info"
            )
            copy_file(
                REPO_BASE / f"configs/example_{_inst_type}.yml",
                cfg_dir / f"{_inst_type}.yml",
                ml_user,
                ml_group,
                cfg_create_mode,
            )
        else:
            logger.warning(
                f"Config directory '{cfg_dir}' does not exist, skipping creation of {_inst_type} config file..."
            )


def do_install():
    global _ENV

    _inst_type = "server"
    check_backup(_inst_type)

    logger.info(f"Installing '{_inst_type}' specific files...")
    copy_file(
        EXAMPLES_DIR / "mlapi.py",
        data_dir / "bin/mlapi.py",
        ml_user,
        ml_group,
        0o755,
    )
    test_msg(
        f"Creating symlinks for ZoMi Server: '/usr/local/bin/mlapi' will symlink to '{data_dir}/bin/mlapi.py'",
    )

    _dest = Path("/usr/local/bin/mlapi")
    if not testing:
        if _dest.exists():
            if interactive:
                x = input(
                    f"The symlink {_dest} already exists, would you like to remove it and create a new one? [Y/n] "
                )
                if x.casefold() == "n":
                    logger.error(
                        f"The symlink {_dest} already exists, Please remove it and try again."
                    )
                    sys.exit(1)
                _dest.unlink()
            else:
                if not force:
                    raise FileExistsError(
                        f"{_dest.name} symlink already exists at {_dest.as_posix()}, please remove and try again."
                    )
                else:
                    _dest.unlink()
                    _dest.symlink_to(f"{data_dir}/bin/mlapi.py")
        else:
            _dest.symlink_to(f"{data_dir}/bin/mlapi.py")
    else:
        if _dest.exists():
            if not force:
                logger.error(f"The symlink {_dest.as_posix()} already exists, Please remove it before actually installing.")
            else:
                test_msg(
                    f"Would have created symlink: '{_dest.as_posix()}' -> '{data_dir}/bin/mlapi.py'"
                )

        else:
            test_msg(
                f"Would have created symlink: '{_dest.as_posix()}' -> '{data_dir}/bin/mlapi.py'"
            )

    create_("secrets", cfg_dir / "secrets.yml")
    create_(_inst_type, cfg_dir / f"{_inst_type}.yml")

    if not create_venv(create_pip_cmd()):
        raise RuntimeError("Failed to create virtual environment!")


def create_pip_cmd() -> List:
    _src: str = (
        f"{REPO_BASE.expanduser().resolve().as_posix()}"
    )
    _cmd_array: List[str] = [
        "-m",
        "pip",
        "install",
    ]
    if args.no_cache:
        logger.info("Disabling pip cache...")
        _cmd_array.append("--no-cache-dir")

    if testing and not editable:
        _cmd_array.append("--dry-run")
    elif testing and editable:
        logger.warning(
            "TESTING: skipping --editable flag, this will not work in test mode"
        )
    elif editable and not testing:
        logger.info(
            "Installing in pip --editable mode, DO NOT remove the source git"
            " directory after install! git pull will update the installed package"
        )
        _cmd_array.append("--editable")

    _cmd_array.append(_src)
    return _cmd_array


def create_venv(cmd_array: List[str]):
    global venv_dir
    if not venv_dir:
        logger.error("VENV directory not set!")
        return False
    venv_dir = venv_dir.expanduser().resolve() if not venv_dir.is_absolute() else venv_dir

    if venv_dir.exists():
        if not force:
            logger.error(f"VENV directory {venv_dir} already exists! Please remove it and try again!")
            return False
        else:
            logger.warning(f"VENV directory {venv_dir} already exists, overwriting...")
            shutil.rmtree(venv_dir)

    _venv = ZoMiEnvBuilder(
        with_pip=True, cmd=cmd_array, upgrade_deps=True, prompt=VENV_NAME
    )
    test_msg(f"Creating \"{VENV_NAME}\" virtual environment in directory: {venv_dir}")
    try:
        _venv.create(venv_dir)
    except FileNotFoundError as e:
        if not testing:
            logger.warning(f"Issue while creating VENV: {e}")
        return False
    except Exception as e:
        logger.exception(f"Failed to create VENV!")
        return False

    _f: Path = data_dir / "bin/mlapi.py"
    test_msg(
        f"Modifying {_f.as_posix()} to use VENV shebang: #!{_venv.context.env_exec_cmd}"
    )
    if not testing:
        if not _f.is_absolute():
            _f = _f.expanduser().resolve()
        content: Optional[str] = _f.read_text() or None
        try:
            with _f.open("w+") as f:
                f.write(f"#!{_venv.context.env_exec_cmd}\n{content}")
        except Exception as e:
            logger.exception(f"Failed to modify {_f}, MAKE SURE TO MODIFY IT YOURSELF WITH the above shebang!")
    return True


class Envsubst:
    strict: bool
    env: Optional[Dict]

    def __init__(self):
        """
        Substitute environment variables in a string. Instantiate and pass a string to the sub method.
        Modified to allow custom environment mappings, and to allow for strict mode
        (only use the custom environment mapping).

        Possibly add dotenv support?.
        """
        self.strict = False
        self.env = None

    def sub(self, search_string: str, env: Optional[Dict] = None, strict: bool = False):
        """
        Substitute environment variables in the given string, allows for passing a custom environment mapping.
        The default behavior is to check the custom environment mapping , the system environment and finally the
        specified default ("somestring" in the examples). If strict is True, the system environment will not be
        checked after the custom environment mapping but, the default will still be used (if needed).

        The following forms are supported:

        Simple variables - will use an empty string if the variable is unset and strict is true
          $FOO

        Bracketed expressions
          ${FOO}
            identical to $FOO
          ${FOO:-somestring}
            uses "somestring" if $FOO is unset, or is set and empty
          ${FOO-somestring}
            uses "somestring" only if $FOO is unset
        """
        self.strict = strict
        if strict:
            logger.info(f"envsubst:: strict mode: {self.strict}")
        if env:
            self.env = env
        # handle simple un-bracketed env vars like $FOO
        a: str = _simple_re.sub(self._repl_simple_env_var, search_string)
        logger.debug(f"envsubst DBG>>> after simple sub {type(a) = }")
        # handle bracketed env vars with optional default specification
        b: str = _extended_re.sub(self._repl_extended_env_var, a)
        return b

    def _resolve_var(self, var_name, default=None):
        if not self.strict and default is None:
            # Instead of returning an empty string in strict mode,
            # return the variable name formatted for substitution
            default = f"${{{var_name}}}"

        if self.env:
            if not self.strict:
                return self.env.get(var_name, os.environ.get(var_name, default))
            else:
                return self.env.get(var_name, default)

        return os.environ.get(var_name, default)

    def _repl_simple_env_var(self, m: re.Match):
        var_name = m.group(1)
        return self._resolve_var(var_name, "" if self.strict else None)

    def _repl_extended_env_var(self, m: re.Match):
        if m:
            # ('ML_INSTALL_DATA_DIR', None, None, None)
            var_name = m.group(1)
            default_spec = m.group(2)
            if default_spec:
                default = m.group(4)
                default = _simple_re.sub(self._repl_simple_env_var, default)
                if m.group(3) == ":-":
                    # use default if var is unset or empty
                    env_var = self._resolve_var(var_name)
                    if env_var:
                        return env_var
                    else:
                        return default
                elif m.group(3) == "-":
                    # use default if var is unset
                    return self._resolve_var(var_name, default)
                else:
                    raise RuntimeError("unexpected string matched regex")
            else:
                return self._resolve_var(var_name, "" if self.strict else None)


def envsubst(string: str, env: Optional[Dict] = None, strict: bool = False):
    """Wraps Envsubst class for easier use.

    Args:
        string (str): String to substitute
        env (Optional[Dict], optional): Custom environment mapping. Defaults to None.
        strict (bool, optional): If True, only use the custom environment mapping. Defaults to False.
    """
    _ = Envsubst()
    return _.sub(string, env=env, strict=strict)


def create_(_type: str, dest: Path):
    src = REPO_BASE / f"configs/example_{_type}.yml"
    envsubst_out = envsubst(src.read_text(), _ENV)
    test_msg(f"Writing {_type.upper()} file to {dest.as_posix()}")
    if not testing:
        dest.expanduser().resolve().write_text(envsubst_out)


def in_venv():
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


class ZoMiEnvBuilder(venv.EnvBuilder):
    """
    Venv builder for ZoMi ML

    :param cmd: The pip command as an array to install ZoMi ML.
    """

    lp: str = "EnvBuilder:"
    install_cmd: List[str]

    def __init__(self, *args, **kwargs):
        self.context: Optional[SimpleNamespace] = None
        self.install_cmd: Optional[List[str]] = kwargs.pop("cmd", None)
        assert isinstance(
            self.install_cmd, list
        ), f"{self.lp} cmd must be a list not {type(self.install_cmd)}"
        assert self.install_cmd, f"{self.lp} cmd must not be empty!"
        super().__init__(*args, **kwargs)

    def post_setup(self, context):
        """
        Set up any packages which need to be pre-installed into the
        environment being created.
        :param context: The information for the environment creation request
                        being processed.
        """
        self.context = context
        old_env = os.environ.get("VIRTUAL_ENV", None)
        os.environ["VIRTUAL_ENV"] = context.env_dir
        self.install_zomi(context)
        if old_env:
            os.environ["VIRTUAL_ENV"] = old_env
        else:
            os.environ.pop("VIRTUAL_ENV")

    def install_zomi(self, context):
        """
        Install ZoMi in the environment.
        :param context: The information for the environment creation request
                        being processed.
        """

        def _popen(_args: List[str]):
            try:
                ran = subprocess.Popen(
                    _args,
                    stdout=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"{self.lp} Error installing zomi-server!")
                logger.error(e)
                if e.stderr:
                    logger.error(e.stderr)
                if e.stdout:
                    logger.info(e.stdout)
                raise e
            else:
                if ran:
                    msg = ""
                    for c in iter(lambda: ran.stdout.read(1), b""):
                        sys.stdout.buffer.write(c)
                        # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe2 in position 0: unexpected end of data
                        # deal with unexpected end of data
                        try:
                            c = c.decode("utf-8")
                        except UnicodeDecodeError:
                            pass
                        else:
                            # add to the msg until we get a newline
                            if c == "\n":
                                logger.info(f"{self.lp} {msg}")
                                msg = ""
                            else:
                                msg += c

        _args = [
            context.env_exec_cmd,
            "-m",
            "pip",
            "install",
        ]
        if args.no_cache:
            _args.append("--no-cache-dir")
        # TORCH
        # if gpu, install proper torch and torchvision -
        # Cuda 11.8 = pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        # Cuda 12.1 = pip3 install torch torchvision
        # ROCm6.0 = pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
        # cpu =  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        _a = list(_args)
        if cpu_only:
            logger.info(f"\n\n{self.lp} Installing CPU TORCH packages...")
            _a.append("--index-url")
            _a.append("https://download.pytorch.org/whl/cpu")
        elif gpu_args:
            logger.info(
                f"\n\n{self.lp} Installing GPU TORCH packages based on {gpu_args}..."
            )

            if gpu_args == ["cuda12.1"]:
                pass
            elif gpu_args == ["rocm6.0"]:
                _a.append("--index-url")
                _a.append("https://download.pytorch.org/whl/rocm6.0")
            elif gpu_args == ["cuda11.8"]:
                _a.append("--index-url")
                _a.append("https://download.pytorch.org/whl/cu118")
            else:
                logger.warning(
                    f"{self.lp} Unknown GPU args: {gpu_args}, using 'cuda12.1'"
                )

        _a.append("torch")
        _a.append("torchvision")
        _popen(_a)

        # ONNX Runtime
        _a = list(_args)
        if cpu_only:
            logger.info(f"\n\n{self.lp} Installing onnxruntime [CPU]...")
            _a.append("onnxruntime")
        elif gpu_args:
            logger.info(f"\n\n{self.lp} Installing onnxruntime [GPU]...")
            _a.append("onnxruntime-gpu")
        _popen(_a)

        # Install ZoMi Server
        self.install_cmd.insert(0, context.env_exec_cmd)
        logger.debug(f"\n\n{self.lp} Installing ZoMi Server: '{self.install_cmd}'")
        _popen(self.install_cmd)

        if opencv:
            _a = list(_args)
            logger.info(f"\n\n{self.lp} Installing opencv-contrib-python-headless [CPU only / No GUI libs]...")
            _a.append("opencv-contrib-python-headless")
            _popen(_a)

        # Install ageitgey/face_recognition
        if args.face_rec:
            logger.info(f"\n\n{self.lp} Attempting to install ageitgey/face-recognition, "
                        f"this may take some time if DLib is not built and installed...")
            _a = list(_args)
            _a.append("face-recognition")
            _popen(_a)




def check_python_version(maj: int, min_: int):
    if sys.version_info.major < maj:
        logger.error(f"Python {maj}+ is required to run this install script!")
        sys.exit(1)
    elif sys.version_info.major == maj:
        if sys.version_info.minor < min_:
            logger.error(f"Python {maj}.{min_}+ is required!")
            sys.exit(1)


if __name__ == "__main__":
    # check python is 3.8+ only
    check_python_version(3, 8)

    # parse args first
    args = parse_cli()

    logger.info(f"Starting install script...")
    if not check_imports():
        msg = f"Missing python dependencies, exiting..."
        logger.critical(msg)
        sys.exit(1)
    else:
        logger.info("All python dependencies that this install script requires have been found.")

    class TqdmLoggingHandler(logging.Handler):
        def __init__(self, level=logging.NOTSET):
            super().__init__(level)

        def emit(self, record):
            from tqdm.auto import tqdm

            try:
                msg = self.format(record)
                # msg = f"TQDM::>>> {msg}"
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)

    tqdm_handler: TqdmLoggingHandler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(log_formatter)
    logger.removeHandler(console)
    logger.addHandler(tqdm_handler)
    models: List[str] = []
    testing: bool = args.test
    editable: bool = args.pip_install_editable
    venv_only: bool = args.venv_only
    force: bool = args.force
    debug: bool = args.debug
    if in_venv():
        logger.info(
            "Detected to be running in a virtual environment, "
            "be aware this install script creates "
            "a venv for ZoMi to run in!"
        )

    install_log = args.install_log
    file_handler = logging.FileHandler(install_log, mode="w")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    if debug:
        logger.setLevel(logging.DEBUG)
        for _handler in logger.handlers:
            _handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled!")
    if testing:
        logger.warning(">>>>>>>>>>>>> Running in test/dry-run mode! <<<<<<<<<<<<<<<<")
        # make sure it is noticed
        time.sleep(2.5)
    models = args.models
    no_models: bool = args.no_models
    force_models: bool = args.force_models
    system_create_mode: int = args.system_create_permissions
    cfg_create_mode: int = args.config_create_permissions
    interactive: bool = args.interactive
    data_dir: Path = args.data_dir
    cfg_dir: Path = args.config_dir
    log_dir: Path = args.log_dir
    tmp_dir: Path = args.tmp_dir
    venv_dir: Path = data_dir / "venv"
    route_host = args.route_host
    route_port = args.route_port
    route_sign_key = args.route_key
    if not route_host:
        route_host = "0.0.0.0"
    if not route_port:
        route_port = "5000"
    if not route_sign_key:
        import secrets

        logger.info("No JWT sign key configured! Generating pseudo-random JWT signature key...")
        route_sign_key = "".join(
            secrets.choice(string.ascii_letters + string.digits) for _ in range(128)
        )
    gpu_args: Optional[List[str]] = args.gpu
    opencv: bool = args.opencv
    cpu_only: bool = args.cpu
    model_dir: Optional[Path] = data_dir / "models"
    data_dir: Optional[Path]
    cfg_dir: Optional[Path]
    log_dir: Optional[Path]
    tmp_dir: Optional[Path]
    args.data_dir = data_dir = data_dir.expanduser().resolve()
    args.config_dir = cfg_dir = cfg_dir.expanduser().resolve()
    args.log_dir = log_dir = log_dir.expanduser().resolve()
    args.tmp_dir = tmp_dir = tmp_dir.expanduser().resolve()
    ml_user, ml_group = args.ml_user or "", args.ml_group or ""
    do_web_user()
    if not ml_user:
        logger.error(
            "user not specified/cant be computed (try to specify user and group), exiting..."
        )
        sys.exit(1)
    args.ml_user = ml_user
    args.ml_group = ml_group

    show_config(args)

    if venv_only:
        logger.info("--venv-only flag supplied, only creating virtual environment...")
        if not create_venv(create_pip_cmd()):
            logger.critical("Failed to create virtual environment!")
            exit(1)
        logger.info("Virtual environment created successfully!")
        exit(0)

    _ENV = {
        "ML_INSTALL_DATA_DIR": data_dir.as_posix(),
        "ML_INSTALL_VENV_DIR": venv_dir.as_posix(),
        "ML_INSTALL_CFG_DIR": cfg_dir.as_posix(),
        "ML_INSTALL_LOGGING_DIR": log_dir.as_posix(),
        "ML_INSTALL_TMP_DIR": tmp_dir.as_posix() if tmp_dir else None,
        "ML_INSTALL_MODEL_DIR": model_dir.as_posix() if model_dir else None,
        "ML_INSTALL_IMAGE_DIR": (data_dir / "images").as_posix(),
        "ML_INSTALL_LOGGING_LEVEL": "debug",
        "ML_INSTALL_LOGGING_CONSOLE_ENABLED": "yes",
        "ML_INSTALL_LOGGING_FILE_ENABLED": "no",
        "ML_INSTALL_LOGGING_SYSLOG_ENABLED": "no",
        "ML_INSTALL_LOGGING_SYSLOG_ADDRESS": "/dev/log",
        "ML_INSTALL_SERVER_ADDRESS": route_host,
        "ML_INSTALL_SERVER_PORT": route_port,
        "ML_INSTALL_SERVER_SIGN_KEY": route_sign_key,
    }

    if args.all_models:
        models = [str(x) for x in available_models.keys()]
        logger.info(f"ALL MODELS requested: Using all available models: {models}")
    if args.model_dir:
        model_dir = Path(args.model_dir)
        logger.info(f"Using model directory: {model_dir}")
    else:
        logger.info(f"No model directory specified, using default: {data_dir}/models")
        model_dir = Path(f"{data_dir}/models")
    if args.env_file:
        parse_env_file(args.env_file)
    if args.models_only:
        logger.info(f"\n***Only installing models! ***\n")
        get_models()
        sys.exit(0)
    if args.config_only or args.secrets_only:
        secrets = False
        _cfg = False
        if args.config_only:
            _cfg = True
        if args.secrets_only:
            secrets = True

        if secrets and _cfg:
            logger.info(f"\n***Only installing config and secrets files! ***\n")
            create_("secrets", cfg_dir / "secrets.yml")
            create_("server", cfg_dir / "server.yml")
        elif secrets:
            logger.info(f"\n***Only installing secrets file! ***\n")
            create_("secrets", cfg_dir / "secrets.yml")
        elif _cfg:
            logger.info(f"\n***Only installing config file! ***\n")
            create_("server", cfg_dir / "server.yml")
        sys.exit(0)

    main()
    if THREADS:
        if any([t.is_alive() for t in THREADS.values()]):
            logger.info(
                "There is at least 1 active thread, waiting for them to finish..."
            )
            for t_name, t in THREADS.items():
                if t.is_alive():
                    logger.debug(f"Waiting for thread '{t_name}' to finish...")
                    t.join()
                    logger.debug(f"Thread '{t_name}' finished\n")
        logger.info("All threads finished")

    logger.info(f"\n\nFinished install script!!\n")
