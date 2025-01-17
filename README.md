# Short-Term Forecast with MLOps
This repo is a tool for structuring, transforming data science project. It is predicting short term forecast for behaviral change. The project incorporates MLOps best practices and focuses on developing models for short-term forecasting.

# MLOps: MLFlow Tracker

The mlflow tracker module provides a wrapper class to simplify the use of MLFlow's tracking functionalities. It:

* Initializes MLFlow runs.
* Logs parameters, metrics, and artifacts.
* Facilitates experiment management and reproducibility.

## Data

The `data/` folder contains scripts related to process time-series data. 
`temporalgnn_data.py` includes a `Dataset` class that can process load and generation timeseries for the purpose of training a graph-based forecasting model using `torch_geometric_temporal`.


## Models
The ``models/`` directory contains methodology used for forecasting the short term electricity consumption in an hourly resolution.

`NeuralProphet` is built on PyTorch and combines Neural Networks and traditional time-series algorithms, inspired by Facebook Prophet and AR-Net.

`temporalgnn_model.py` includes the pytorch_lightning implementation of the A3TGN2 model (which combines graph convolutional networks (GCN) and gated recurrent units (GRU)) for training, validation, and prediction.