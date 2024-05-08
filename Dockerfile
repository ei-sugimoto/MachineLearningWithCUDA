FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN git config --global user.email "eikun1014@gmail.com" && git config --global user.name "ei-sugimoto"

RUN pip install -y torch_geometric