#!/bin/bash
sudo apt install python3-psutil python3-tqdm python3-venv python3-pip python3-setuptools python3-wheel python3-distro python3-requests
sudo examples/install.py --cpu --user www-data --group www-data --gpu cuda12.1 --editable --interactive --opencv --debug
