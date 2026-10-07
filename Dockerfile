FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUTF8=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY predict.py /workspace/predict.py
COPY pkgs /workspace/pkgs
COPY weights /workspace/weights

ENTRYPOINT ["python", "/workspace/predict.py", "--input", "/input", "--output", "/output", "--weights", "/workspace/weights"]
