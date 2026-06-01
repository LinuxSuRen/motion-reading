FROM registry.cn-hangzhou.aliyuncs.com/linuxsuren/python:3.11-slim

RUN rm -f /etc/apt/sources.list.d/debian.sources && \
    echo "deb https://mirrors.aliyun.com/debian/ trixie main" > /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian/ trixie-updates main" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian-security/ trixie-security main" >> /etc/apt/sources.list

RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && \
    apt-get -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false update && \
    apt-get -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libegl1 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && \
    pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r requirements.txt

COPY server.py pose_engine.py ./
COPY robot_arms/ robot_arms/
COPY static/ static/

EXPOSE 8000

CMD ["python", "-u", "server.py"]
