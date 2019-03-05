FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         wget \
         git \
         curl \
         vim \
         gcc g++ \
         ca-certificates \
         libjpeg-dev \
         zsh \
         libcudnn7-dev=`dpkg -s libcudnn7 | grep Version | awk '{print $2};'` \
         libnccl-dev=`dpkg -s libnccl2 | grep Version | awk '{print $2};'` \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/* 

RUN touch ~/.zshrc && wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true

RUN sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="ys"/' ~/.zshrc &&\
     chsh -s /bin/zsh

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -c conda-forge ipdb && \
     /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include tqdm tensorflow tensorboard scikit-learn && \
     /opt/conda/bin/pip install tensorboardX &&\
     /opt/conda/bin/conda install pytorch torchvision cudatoolkit=9.0 -c pytorch &&\
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

#RUN git clone --branch master --recursive https://github.com/pytorch/pytorch.git
#RUN cd pytorch && mkdir build && cd build \
#    && cmake .. \
#    -DCUDA_ARCH_NAME=Manual \
#    -DCUDA_ARCH_BIN="60 61 70" \
#    -DCUDA_ARCH_PTX="61 70" \
#    -DUSE_NNPACK=OFF \
#    -DUSE_ROCKSDB=OFF \
#    && make -j"$(nproc)" install \
#    && ldconfig \
#    && make clean \
#    && cd .. \
#    && rm -rf build

ENV PYTHONPATH /usr/local

RUN /opt/conda/bin/pip install --no-cache-dir git+https://github.com/pytorch/tnt.git@master pretrainedmodels
RUN git clone https://www.github.com/nvidia/apex && \
	cd apex && \
	/opt/conda/bin/python setup.py install && \
	cd .. && \
	rm -rf apex
# This must be done before pip so that requirements.txt is available
WORKDIR /code
RUN chmod -R a+w /code
RUN mkdir /datadrive
