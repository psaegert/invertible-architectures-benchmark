# syntax=docker/dockerfile:1

# BayesFlow 2.0.14 supports Python 3.12/3.13, while the local Windows venv
# uses Python 3.14. Use an explicit supported Python version for all families.
FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    KERAS_BACKEND=torch \
    MPLBACKEND=Agg

WORKDIR /app

# git is needed for the pinned FrEIA commit; libgomp supports numerical wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CUDA user-space libraries arrive with the pinned PyTorch wheel dependencies.
# The host supplies the NVIDIA driver through Docker's --gpus option.
COPY requirements-docker.lock ./requirements-docker.lock
# Keep downloaded wheels in the builder cache, outside the resulting image.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install pip==26.0.1 \
    && python -m pip install -r requirements-docker.lock \
    && python -m pip check

# Triton compiles C helpers at runtime, including some eager CUDA backward
# paths in PyTorch. Keep the compiler after the large dependency layer so
# toolchain changes can reuse already installed GPU packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy source directories explicitly; unrelated root files stay out of the image.
COPY data ./data
COPY models ./models
COPY experiments ./experiments
COPY metrics ./metrics
COPY robot_ik ./robot_ik
COPY tests ./tests
COPY pytest.ini requirements.txt requirements-docker.in README.md ./

# Also allow creating new run/result directories without a host mount.
# Mount the project at /app for persistent, shared train/evaluation outputs.
RUN mkdir -p experiments/runs metrics/results metrics/cache docs output

# Starting the image never launches training or evaluation automatically.
# Override with e.g. `python experiments/train_fif.py --help`.
CMD ["python", "--version"]
