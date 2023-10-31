# Holistic and Lightweight Approach for Solar Irradiance (GHI) Forecasting

With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript:

> M. Jain, P. Yadav and S. Dev, "Holistic and Lightweight Approach for Solar Irradiance Forecasting," in IEEE Transactions on Geoscience and Remote Sensing, 2023 [UNDER REVIEW].

---

This work presents a novel framework for GHI forecasting that leverages meteorological variables, historical GHI data, GSI images, and satellite-derived cloud masks. By condensing image data into lower-dimensional feature vectors, this framework accommodates for longer historical contexts for forecasting GHI over a $60$-minute horizon. The complete framework is outlined in the figure below.
[![GSI Forecasting Framework](/imgs/GSIreducer.png)](/imgs/GSIreducer.pdf)

## Purpose
In the absence of suitable training data to directly achieve this transformation, our approach involves using the $\texttt{civ}$ vector to construct a model for estimating global horizontal irradiance (GHI) in subsequent processing steps. This model takes into account key environmental parameters such as solar zenith angle (SZA), solar azimuth angle (SAA), and the clear sky model (CSM) output. The overall architecture is depicted in the figure below.



## Core Scripts

In this enterprise, we utilize several core scripts to achieve the goals of our project. Here's a brief description of each of these essential scripts:

1. **`script.py`**:
   - Main code file responsible for training the model.
   - Specifies the initialization arguments required for model training.

2. **`trainFromFeatures.py`**:
   - Contains the training loop that supports both single and multi-GPU execution.
   
3. **`evaluate.py`**:
   - This script plays a critical role in assessing the model's performance.
   - It calculates and provides the following evaluation metrics:
     - Root Mean Square Error (RMSE)
     - Mean Absolute Error (MAE)
     - Coefficient of Determination ($R^2$)
   - Additionally, it generates GHI prediction curves for a specific date, aiding in the visualization of model predictions.

4. **`createReducedData.py`**:
   - This script takes advantage of the trained model to create the cloud impact vector ($\texttt{cvi}$) for a given year. It stores these vectors as Numpy objects along with their associated timestamps, facilitating further analysis and downstream applications.

4. **`models/sirtaGSIGHImodel.py`**:
   - This script contains the PyTorch model, defining the architecture used in the project.

5. **`datasets/sirtaGSIGHI.py`**:
   - It's responsible for creating the PyTorch dataset required for training and evaluation.

6. **`utils/*`**:
   - The `utils` directory houses various utility scripts that perform essential calculations for the project.
   - Included in this directory are scripts for computing values such as Solar Zenith Angle (SZA), Solar Azimuth Angle (SAA), and Clear Sky Model (CSM) output.

Please note that the detailed implementation of creating $\texttt{cf}$ and $\texttt{cc}$ from the raw GSI images can be found in [this](https://github.com/pyaada/sky-cloud-fraction.git) GitHub repository.