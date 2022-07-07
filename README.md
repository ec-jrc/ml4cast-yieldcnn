# Lean Temporal Convolutional Neural Network for yield forecasting

Training temporal Convolution Neural Networks (CNNs) on satellite image time series for yield forecasting.

### Table of contents
* [Prerequisites](#prequisites)
* [Technologies](#technologies)
* [Setup](#setup)


![Model architecture](figures/yieldcnn_architecture.png)


## Prerequisites
To set up the environment:

```
git clone https://github.com/ec-jrc/ml4cast-be-yieldcnn.git
cd ml4cast-be-yieldcnn
conda env create -f environment.yml
conda activate leanyf
```

If a GPU is available, then type in:
```pip install tensorflow-gpu==2.3.0```

You can monitor GPU usage with the following command: 
```watch -n 1 nvidia-smi```

## Directory tree structure

Below is the directory tree structure of the project. To run the project, one only needs the ```raw_data``` folder to be 
populated. Other folders are created automatically.

```
leanyf
├── data
│       ├── meta
│       └── params
│           └── Archi_{model_name}_{normalisation}_{input_size}
│                └── crop_{index}
│                    ├── month_{forecast}
│                        └── best_model
├── figures
└── raw_data
```

Aggregated results are stored in ```data/model_evaluation_{input_data}CNN.csv```.

## Script overview

As a helper, certain variables have been hardcoded in ```mysrc/constants.py```. 


Scripts were named as explicitly as possible. Here is a short description of the main ones:
* ```preprocess_1D_inputs.py```: Convert tabular data into arrays usable by a 1D CNN. Arguments are defined in ```mysrc/constants.py```.
* ```preprocess_2D_inputs.py```: Format tabular data into arrays usable by a 2D CNN and save them as ```pickles``` with  the ```dill``` package. It is recommended to re-run this step on the machines where the code will run. 
* ```optimise_so_1D_architectures.py```: Full model optimisation and evaluation for 1D input data.
* ```optimise_so_2D_architectures.py```: Full model optimisation and evaluation for 2D input data.
* ```launcher_1D.sh```: Submit ```optimise_so_1D_architectures.py``` on AWS.
* ```launcher_2D.sh```: Submit ```optimise_so_2D_architectures.py``` on AWS.

Models have been developed in the ```deeplearning``` folder:
* ```architecture_complexity_1D.py```: Architecture definition of 1D models
* ```architecture_complexity_2D.py```: Architecture definition of 1D models
* ```architecture_features.py```: Defining tensorflow.keras architecture, and training the models



## Contributors
 - [Dr. Franz Waldner](https://scholar.google.com/citations?user=4z2zcXwAAAAJ&hl=en&oi=ao)
 - [Dr. Michele Meroni](https://scholar.google.com/citations?user=iQk-wj8AAAAJ&hl=en&oi=ao)
 - [Filip Sabo](https://scholar.google.com/citations?user=tSuJbVQAAAAJ&hl=it&oi=ao)


