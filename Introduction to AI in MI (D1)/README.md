# Winter School Day 1: Introduction to AI in medical imaging
This repository contains all the materials for the first day of [AI in Medical Imaging Winter School](https://github.com/RISE-MICCAI/AI-in-Medical-Imaging-Winter-2022-School). The aim of this tutorial is to give a high-level introduction to using AI in medical imaging. We provide an example of a principled workflow starting from managing your dataset in XNAT (a specialized imaging informatics platform for imaging-based research) to image preprocessing and machine learning model application.

This tutorial consists of three parts:
- High-level introduction to XNAT
- Two Jupyter notebooks:
	- XNAT and image processing
	- Using XNAT in machine learning workflows

# Installation instructions
In order to run this tutorial, you have to use Python 3.7+. We recommend to install and run this tutorial locally using a Python virtual environment. We suggest using [virtualenv](https://virtualenv.pypa.io/en/latest/index.html) for managing virtual environments. Installation instructions for virtualenv can be found [here](https://virtualenv.pypa.io/en/latest/installation.html). After installing `virtualenv`, create a virtual environment by running `virtualenv venv` and [activate it](https://virtualenv.pypa.io/en/latest/user_guide.html#activators). Finally, run `pip install -r requirements.txt` to install all requirements needed for running the Jupyter notebooks. Confirm that the installation worked correctly by running `jupyter notebook` in the terminal.

This tutorial can also be installed when using `conda` distributions: `conda install --file requirements.txt`.

# Running the tutorial
First, open the provided pdf file and follow the instructions there. During this part you will become familiar with XNAT - a powerful platform for imaging-based research. After you have finished all the tasks listed there, open the first Jupyter notebook - XNAT and Image preprocessing. Here we discuss how to work with XNAT from software and perform image preprocessing. When you are done with the first notebook, open the second one: Using XNAT in machine learning workflows. In this final part of the tutorial, we show how you could integrate data management practices into your machine learning research workflows.

# License information
The code and data used in this tutorial can only be used by the participants of AI in Medical Imaging Winter School. The data (including pretrained machine learning model) can only be used by the participants for the duration of the winter school.

# Acknowledgements
We would like to thank dr. Bo Li and dr. Esther Bron for their help with creating this tutorial and providing the pretrained segmentation model.