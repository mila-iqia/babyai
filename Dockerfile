# To build this docker image:
# sudo docker build .
#
# To run the image:
# sudo nvidia-docker run -it <image_id>

# Note: a more recent nvidia/cuda image means users must
# have a more recent CUDA install on their systems
#FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# Install dependencies
RUN apt-get update -y
RUN apt-get install -y qt5-default qttools5-dev-tools git python3-pip
RUN pip3 install --upgrade pip

# Clone and install BabyAI git repo
RUN git clone https://github.com/mila-udem/babyai.git
WORKDIR babyai
RUN pip3 install --editable .

# Copy models into the docker image
COPY models models/
