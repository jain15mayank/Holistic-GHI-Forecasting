# GHI Forecasting Model
This component aims to utilize the encoded representation, of satellite-derived cloud masks and the GSI cloud images, as features alongside the relevant meteorological variables, historical GHI data, SZA, SAA, and CSM values to forecast GHI over a $60$- minute horizon.

LSTM-based model (shown in figure below) and PatchTST <a href="#ref1">[1]</a> were implemented to make GHI forecasts.

<div style="text-align:center;">
   <a href="../imgs/TSforecastingModels-LSTM.pdf">
      <img src="../imgs/TSforecastingModels-LSTM.png" alt="LSTM-based GHI Forecasting Model" width="400">
   </a>
</div>

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

## References & Acknowledgement

1. <a id="ref1"></a> [Y. Nie, N. H. Nguyen, P. Sinthong and J. Kalagnanam, "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers," *The Eleventh International Conference on Learning Representations (ICLR), Kigali, Rwanda*, 2023](https://openreview.net/forum?id=Jbdc0vTOcol)