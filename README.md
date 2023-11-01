# Holistic and Lightweight Approach for Solar Irradiance Forecasting

With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript:

> M. Jain, P. Yadav and S. Dev, "Holistic and Lightweight Approach for Solar Irradiance Forecasting," in IEEE Transactions on Geoscience and Remote Sensing, 2023 [UNDER REVIEW].

---

## A Novel Framework for GHI Forecasting

This work introduces an innovative framework for global horizontal irradiance (GHI) forecasting. It leverages a combination of meteorological variables, historical GHI data, ground-based sky imager (GSI) images, and satellite-derived cloud masks. One of its key innovations lies in transforming complex image data into lower-dimensional feature vectors, enabling it to provide longer historical contexts for GHI forecasts, spanning a 60-minute horizon. The proposed framework is visually outlined below:

[![GSI Forecasting Framework](./imgs/framework.png)](./imgs/framework.pdf)

## Key Components of the Framework

1. **Identifying Relevant Meteorological Variables**

    - We employed diverse feature selection strategies to pinpoint the most critical meteorological variables.
    - The corresponding code for this process is accessible in the directory named `Identify Met Vars`.

2. **Optimizing the Clear Sky Model (CSM)**

    - With a myriad of CSMs available in the literature, our objective was to identify the optimal one for a specific geographical location.
    - Recognizing that the optimality of CSM models can vary with time, we adopted a month-wise approach to select the best models.
    - You can find the code related to this optimization in the `Identify CSM` directory.

3. **Sky/Cloud Image Segmentation**

    - In many cases, annotated sky/cloud images are not readily available for training the segmentation model for a given location. To address this, we applied a generative augmentation strategy to enhance the model's generalizability when dealing with out-of-distribution datasets.
    - Once trained, the segmentation model was utilized to segment sky/cloud images for the specific location. These segmented images were then corrected for fisheye distortion, and cloud fraction values were calculated for each grid element after dividing the undistorted image into a $4\times4$ grid.
    - Detailed code for this process is located in the `Cloud Segmentation` directory.

4. **Sky/Cloud Image Classification**

    - Transfer learning techniques were employed to perform sky/cloud image classification using the SWIMCAT dataset.
    - Similar to the segmentation component, the undistorted version of the original image was divided into a $4\times4$ grid, with each element classified into one of the five predefined classes.
    - Code relevant to this classification task can be found in the `Cloud Classification` directory.

5. **Encoding GSI Images**
    - Encoded feature vector or cloud impact vector ($\texttt{civ}$) is constructed from the cloud fraction and cloud classification vectors.
    - Solar zenith angle (SZA) and solar azimuth angle (SAA) were also considered in this step to righteously estimate the $\texttt{civ}$.
    - However, in the absence of suitable training data to directly achieve this transformation, a neural network architecture is designed to estimate GHI instead - which was then trained to extract $\texttt{civ}$ from the intermediate layer.
    - Details about this component and the code relevant to this task can be found in the `GSI Encode` directory.

6. **GHI Forecasting Model**
    - Finally, the relevant meteorological features, encoded satellite-derived cloud masks, encoded GSI images, historical GHI, SZA, SAA, and CSM values were used as features to forecast GHI over the next $60$-minute interval in this component.
    - More details and code relevant to this component can be found in the `GHI Forecasting` directory.