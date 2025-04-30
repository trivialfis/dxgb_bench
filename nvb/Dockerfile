FROM rapidsai/ci-conda:cuda12.8.0-ubuntu22.04-py3.12

ENV DEBIAN_FRONTEND noninteractive
SHELL ["/bin/bash", "-c"]

RUN \
    apt-get update && apt-get upgrade -y

RUN \
    mkdir ws

COPY nvbandwidth ws/nvbandwidth

COPY nvbandwidth.yml /ws/nvbandwidth.yml

RUN --mount=type=cache,target=/opt/conda/pkgs mamba env update -n base -f ws/nvbandwidth.yml -v && mamba clean --all --yes

RUN \
    cd /ws/nvbandwidth && \
    mkdir build && cd build && \
    cmake .. -GNinja && \
    time ninja && \
    cp ./nvbandwidth /usr/bin/nvbandwidth
