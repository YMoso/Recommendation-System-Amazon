FROM jupyter/scipy-notebook:python-3.10

USER root
RUN pip install scikit-surprise

USER jovyan
WORKDIR /home/jovyan/work