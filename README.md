# **An Attention-Based Framework for Integrating WSI and Genomic Data in Cancer Survival Prediction**

This repository contains the code for An Attention-Based Framework for Integrating WSI and Genomic Data in Cancer Survival Prediction. The experiments in this study utilize two datasets: **TCGA-LUAD** and **TCGA-BLCA**, obtained from the [Genomic Data Commons (GDC) Data Portal](https://portal.gdc.cancer.gov/).

## **Data Acquisition**

The TCGA-LUAD and TCGA-BLCA datasets can be downloaded from the [GDC Data Portal](https://portal.gdc.cancer.gov/) by following these steps:

1. Visit the [GDC Data Portal](https://portal.gdc.cancer.gov/).
2. Use the search bar to find **TCGA-LUAD** or **TCGA-BLCA**.
3. Select the required data types (e.g., clinical data, histopathology images, and omics data).
4. Add the selected files to the cart and proceed to download using the GDC Data Transfer Tool or the provided web interface.

Detailed instructions on dataset download and preprocessing can be found in the official GDC documentation.

## **Data Preprocessing**

The data preprocessing follows the same methodology as the [PORPOISE](https://github.com/mahmoodlab/PORPOISE) pipeline, ensuring consistency in feature extraction and data structuring.

### **Histopathology Image Processing**
- Whole-slide images (WSIs) are tiled into non-overlapping patches.
- Background and low-quality patches are filtered out.
- Feature extraction is performed using a pre-trained deep learning model (e.g., ResNet50, CLAM).
- Extracted features are stored in `.pt` files for MIL-based processing.

### **Omics Data Processing**
- RNA-seq and other omics data are normalized and transformed into structured feature matrices.
- Missing values are handled via imputation techniques.
- The processed data are aligned with corresponding histopathology image features for multimodal fusion.

For detailed preprocessing steps, please refer to the PORPOISE repository and adapt the scripts accordingly.

## **Installation**

To set up the environment, install the required dependencies:

```bash
conda env create -f environment.yml
conda activate your_env_name

