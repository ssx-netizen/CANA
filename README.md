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
```

## **Run the Model**

You can run the model using the following command:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python main.py --which_splits 5foldcv --split_dir tcga_blca --mode coattn --model_type multimodal --data_root_dir ./DATA_ROOT_DIR --fusion 'bilinear' --results_dir /root/ssx/dist3 --apply_sig --bag_loss nll_surv --max_epochs 50 --early_stopping >./logs/dist_3.txt 2>&1 &
```

### **Explanation of the command:**

- `CUDA_VISIBLE_DEVICES=0`: Ensures that the computation is run on the specified GPU device (GPU 0).
- `nohup`: Runs the process in the background, allowing it to continue even if the session is closed.
- `python main.py`: Executes the main Python script for training and evaluation.
- `--which_splits 5foldcv`: Specifies the type of cross-validation splits (5-fold cross-validation).
- `--split_dir tcga_blca`: Refers to the directory for the TCGA-BLCA dataset.
- `--mode coattn`: Defines the mode of attention mechanism for multimodal fusion.
- `--model_type multimodal`: Specifies the model type to be used, in this case, a multimodal approach.
- `--data_root_dir ./DATA_ROOT_DIR`: Specifies the root directory of the dataset.
- `--fusion 'bilinear'`: Defines the fusion strategy (bilinear).
- `--results_dir /root/ssx/dist3`: Directory to store the results.
- `--apply_sig`: A flag for applying the sigmoid function.
- `--bag_loss nll_surv`: Specifies the loss function (negative log-likelihood for survival analysis).
- `--max_epochs 50`: Limits the training to a maximum of 50 epochs.
- `--early_stopping`: Activates early stopping to prevent overfitting.

