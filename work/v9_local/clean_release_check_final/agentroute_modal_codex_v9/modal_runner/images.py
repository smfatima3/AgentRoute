"""Image definitions only. Builds occur through authenticated Modal calls."""
from pathlib import Path
import modal

CONTROL=Path(__file__).resolve().parents[1]

def images(package,provided_upstream=None):
    package=Path(package)
    gpu=(modal.Image.from_registry('nvidia/cuda:12.4.1-devel-ubuntu22.04',add_python='3.12')
        .apt_install('git','build-essential')
        .add_local_file(package/'requirements/serving.lock','/opt/serving.lock',copy=True)
        .run_commands('python -m venv /opt/gpu','/opt/gpu/bin/python -m pip install -r /opt/serving.lock')
        .add_local_dir(package,'/opt/package',copy=True,ignore=['**/__pycache__/**'])
        .add_local_dir(CONTROL/'modal_runner','/opt/control/modal_runner',copy=True,ignore=['**/__pycache__/**'])
        .env({'HF_HOME':'/hf','PYTHONUNBUFFERED':'1','CUDA_HOME':'/usr/local/cuda'}))
    cpu=(modal.Image.debian_slim(python_version='3.12')
        .apt_install('git','docker.io','build-essential')
        .add_local_file(package/'requirements/client.lock','/opt/client.lock',copy=True)
        .run_commands('python -m venv /opt/client',
            '/opt/client/bin/python -m pip install -r /opt/client.lock')
        .add_local_file(package/'pyproject.toml','/opt/client-package/pyproject.toml',copy=True)
        .add_local_dir(package/'agentroute','/opt/client-package/agentroute',copy=True,ignore=['**/__pycache__/**'])
        .run_commands('/opt/client/bin/python -m pip install --no-deps /opt/client-package',
            '/opt/client/bin/python -m pip check')
        .add_local_dir(package,'/opt/package',copy=True,ignore=['**/__pycache__/**'])
        .add_local_dir(CONTROL/'modal_runner','/opt/control/modal_runner',copy=True,ignore=['**/__pycache__/**'])
        .env({'HF_HOME':'/hf','PYTHONUNBUFFERED':'1'}))
    if provided_upstream:
        cpu=cpu.add_local_dir(Path(provided_upstream),'/opt/provided_upstream',copy=True)
    return cpu,gpu
