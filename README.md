# Glacier Seismology Project

This project focuses on the detection and classification of microseismic events in glacial environments using signal processing and machine learning techniques.


## Objectives

- Process continuous seismic data recorded on glaciers
- Detect microseismic events
- Classify signals using machine learning methods
- Contribute to hazard monitoring and early warning systems


## Project Structure

project/
│
├── data/          # raw and processed data
├── notebooks/     # exploratory analysis
├── src/           # core algorithms
├── results/       # figures and outputs
├── environment.yml
└── README.md


## Installation

Clone the repository and create the environment:
    $ conda env create -f environment.yml
    $ conda activate glacier-seismo


## Usage

Run Jupyter notebooks:
    jupyter notebook

or run scripts from src/:
    python src/main.py


## Environment Management

Activate the environment:
    $ conda activate glacier-seismo

Deactivate:
    $ conda deactivate

Add a package:
    $ conda install package_name

Update environment file:
    $ conda env export > environment.yml

Clone environment:
    $ conda create --name glacier-seismo-test --clone glacier-seismo

Remove environment:
    $ conda remove --name glacier-seismo --all