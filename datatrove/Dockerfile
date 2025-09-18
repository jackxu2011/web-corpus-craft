FROM python:3.11-slim

WORKDIR /app

# 安装所需依赖（去掉 python3-distutils，替换 libatlas-base-dev）
RUN apt update && apt install -y --no-install-recommends \
    cmake \
    build-essential \
    g++ \
    python3-dev \
    libatlas3-base \
    liblapack-dev \
    aria2 \
 && apt clean \
 && rm -rf /var/lib/apt/lists/* ~/.cache/pip

COPY . /app/datatrove_lang_quality_dedupe
WORKDIR /app/datatrove_lang_quality_dedupe

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
 && pip install --default-timeout=6000 --no-cache-dir -r requirements.txt

ENTRYPOINT ["python"]
CMD ["main.py"]
