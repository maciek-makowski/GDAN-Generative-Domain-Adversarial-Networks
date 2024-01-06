# Use the official Ubuntu base image
FROM ubuntu:20.04

# Update the package list and install necessary dependencies
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
# Install py39 from deadsnakes repository
RUN apt-get install -y python3.9
# Install pip from standard ubuntu packages
RUN apt-get install -y python3-pip

# Set the default Python version to 3.9
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Display the installed Python version
RUN python3 --version

RUN apt-get install -y git 

RUN git clone https://github.com/zykls/whynot.git \
    && cd whynot \
    && pip install -r requirements.txt

WORKDIR /app 

COPY main.py /app

CMD ["python3", "main.py"]