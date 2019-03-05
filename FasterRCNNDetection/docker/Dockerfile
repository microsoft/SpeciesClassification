FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         wget \
         git \
         curl \
         vim \
         ca-certificates \
         zsh \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

RUN touch ~/.zshrc && wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true

RUN sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="ys"/' ~/.zshrc &&\
     echo "python -m apex.parallel.multiproc train.py --fp16" >> ~/.zsh_history &&\
     chsh -s /bin/zsh

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include tqdm cupy scikit-image matplotlib tensorboard && \
     /opt/conda/bin/conda install -y -c conda-forge fire ipdb && \
     /opt/conda/bin/conda install pytorch torchvision cuda91 -c pytorch && \
     /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/pip install --no-cache-dir git+https://github.com/pytorch/tnt.git@master git+https://github.com/lanpa/tensorboard-pytorch

ENV PATH /opt/conda/bin:$PATH
# This must be done before pip so that requirements.txt is available
WORKDIR /code
RUN chmod -R a+w /code
RUN mkdir /datadrive
