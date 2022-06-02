FROM nvcr.io/nvidia/pytorch:21.07-py3

ENV http_proxy http://proxy.pfh.research.philips.com:8080
ENV https_proxy http://proxy.pfh.research.philips.com:8080

RUN apt-get update
COPY requirements.txt .

RUN pip3 install -r requirements.txt

WORKDIR /workdir
COPY main.py config.json mbconv_block.py mbconv_with_se_block.py model.py se_block.py /workdir/