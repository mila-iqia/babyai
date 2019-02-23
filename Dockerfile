# Note: a more recent image means users must have a more recent CUDA install
#FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# install
RUN apt-get update -y
RUN apt-get install -y qt5-default qttools5-dev-tools git python3-pip
RUN pip3 install --upgrade pip

# code
RUN git clone https://github.com/mila-udem/babyai.git
WORKDIR babyai
RUN pip3 install --process-dependency-links --editable .

# copy models
COPY models models/
