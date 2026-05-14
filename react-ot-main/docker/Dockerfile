FROM deepprinciple-cn-beijing.cr.volces.com/public-registry/python:3.10.16

ENV http_proxy http://100.68.166.13:3128
ENV https_proxy http://100.68.166.13:3128
ENV HTTP_PROXY http://100.68.166.13:3128
ENV HTTPS_PROXY http://100.68.166.13:3128

ENV no_proxy mirrors.tuna.tsinghua.edu.cn

COPY . /app

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN pip install --no-cache-dir torch -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN pip install --no-cache-dir torch_geometric -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN pip install --no-cache-dir pytorch-lightning -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN pip install --no-cache-dir torchdiffeq -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN apt-get update && apt-get -y install build-essential python3-dev python3-pip libomp-dev
RUN pip install torch_scatter -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# RUN TMPDIR=/home/ubuntu/tmp/ pip install torch-sparse -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# RUN TMPDIR=/home/ubuntu/tmp/ pip install torch-cluster -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple