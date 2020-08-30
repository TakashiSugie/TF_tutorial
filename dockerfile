FROM tensorflow/tensorflow:latest
#git install時にtime zoneの選択で止まってしまうので先に環境変数を設定
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update  
RUN apt-get install -y python3 python3-pip git cmake libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6
RUN pip3 install  opencv-python matplotlib

WORKDIR /tf_test
COPY copy_dir /tf_test/