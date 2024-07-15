# ZoneMinder Machine Learning API (zomi-server)
>[!CAUTION]
> :warning: This software is in **ALPHA** stage, expect issues and incomplete, unoptimized, janky code ;) :warning:

This is a FastAPI based server component for the new ZoneMinder ML add-on. Please see the 
[workflow docs](docs/workflow.md) for information on how the server processes requests. 
There is also rudimentary support for [color detection](docs/Config/color.md) of cropped bounding boxes.

# Supported ML backends and hardware
The server currently has basic support for several ML backends and hardware accelerators:

**ML backends:**
- OpenCV (DarkNet)
- PyTorch - `torchvision`
- ONNXRuntime (YOLO v8, YOLO-NAS, [YOLO v10 *WIP*])
- TensorRT (YOLO v8, YOLO-NAS, [YOLO v10 *WIP*])
- pycoral (coral.ai Edge TPU)
- openalpr (local binary)
- [face-recognition](https://github.com/ageitgey/face_recognition) based on D-Lib
- HTTP
    - platerecognizer.com
    - AWS Rekognition
- *Open an issue or pull request to get a backend/api supported*

**Hardware**
- CPU
    - OpenVINO is planned 
- Nvidia GPU (CUDA / cuDNN / Tensor RT)
- AMD GPU (Pytorch/onnxruntime ROCm) *WIP / Untested*
- Coral.ai Edge TPU

# Install
Please see the [installation docs](docs/install.md) for more information.

# Swagger UI
>[!TIP]
> Swagger UI is available at the server root: `http://<server>:<port>/`

The server uses FastAPIs built-in Swagger UI which shows available endpoints, response/request schema and 
serves as self-explanatory documentation.

>[!WARNING] 
> Make sure to authorize first! All requests require a valid JWT token. 
> If you haven't enabled auth in the `server.yml` config file, any username:password combo will work.
>![Authorize in Swagger UI](docs/assets/zomi-server_auth-button.png)

# User authentication
>[!CAUTION]
> You can enable and disable authentication, but all requests must have a valid JWT token. When authentication is disabled,
> the login endpoint will accept any username:password combo and supply a valid usable token.

## Default user
The default user is `imoz` with the password `zomi`.

# User Management
User management is done using the `mlapi` script and the `user` sub-command. 
For more information, please see the [User Management](docs/user_management.md) docs.

# Start the server
>[!TIP]
>After installation, there should be a system-wide `mlapi` script command available in `/use/local/bin`

The server can be started with the `mlapi` script.
```bash
mlapi -C /path/to/config/file.yml
```

## Debugging
The server can be started in debug mode with the `--debug` or `-D` flag.
```bash
mlapi -C /path/to/config/file.yml --debug
```

# SystemD service
A SystemD service file example is provided in the [configs/systemd](configs/systemd/mlapi.service) directory.
```bash
sudo cp ./configs/systemd/mlapi.service /etc/systemd/system
sudo chmod 644 /etc/systemd/system/mlapi.service
sudo systemctl daemon-reload
# --now also starts the service while enabling it to run on boot
sudo systemctl enable mlapi.service --now
```

# Fail2Ban
A Fail2Ban filter and jail configuration example is provided in the [configs/fail2ban](configs/fail2ban) directory.

# logrotate
A logrotate configuration file example is provided in the [configs/logrotate](configs/logrotate) directory.

# Docker
Work is being done to create Docker images for the server.