FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y software-properties-common vim
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update

RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN apt-get install -y git

# update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel
RUN apt-get -y update && apt-get -y install ffmpeg
# RUN apt-get -y update && apt-get -y install git wget python-dev python3-dev libopenmpi-dev python-pip zlib1g-dev cmake python-opencv

ENV CODE_DIR /root/code

COPY . $CODE_DIR/nti
WORKDIR $CODE_DIR/nti

# Clean up pycache and pyc files
RUN rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip3 install tensorflow && \
    pip3 install -e .[test]

RUN chmod 777 $CODE_DIR/nti/gymfc/download_gazebo.sh
RUN $CODE_DIR/nti/gymfc/download_gazebo.sh
RUN pip3 install -e $CODE_DIR/nti/gymfc/
RUN pip3 install -r $CODE_DIR/nti/gymfc/examples/requirements.txt
WORKDIR $CODE_DIR/nti/gymfc/gymfc/envs/assets/gazebo/plugins
RUN ./build_plugin.sh
WORKDIR $CODE_DIR/nti

CMD /bin/bash
