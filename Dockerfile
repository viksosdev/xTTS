ARG BASE=nvidia/cuda:11.8.0-base-ubuntu22.04
FROM ${BASE}

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends gcc g++ make python3 python3-dev python3-pip python3-venv python3-wheel espeak-ng libsndfile1-dev && rm -rf /var/lib/apt/lists/*
RUN pip3 install llvmlite --ignore-installed

# Install Dependencies:
RUN pip3 install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
RUN rm -rf /root/.cache/pip

# Copy TTS repository contents:
WORKDIR /root
COPY . /root

EXPOSE 8002

HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=5 CMD curl --fail https://0.0.0.0:8002/health || exit 1

RUN make install

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
