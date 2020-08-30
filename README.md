# TF_tutorial
docker build -t tf_image .

docker -it --name tf_container tf_image:latest /bin/bash
