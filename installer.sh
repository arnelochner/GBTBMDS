#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

conda create -y --name graph36_final
conda activate graph36_final
conda install -y  python=3.6
pip install pyrouge
conda install -y -c conda-forge sentencepiece
conda install -y -c anaconda nltk
conda install -y -c paddle paddlepaddle-gpu="1.8.1 py36_gpu_cuda10.0_many_linux"
conda install -y -c nvidia nccl
conda install -c bioconda perl-xml-libxml 
conda install -c bioconda perl-lwp-protocol-https

cd pyrogue/tools/ROUGE-1.5.5/data
rm WordNet-2.0.exc.db
./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db


