# **Model Training**

This folder contains all the resources needed to train the machine learning model used in the **Netlist Generator** project. It includes datasets, scripts, configuration files, and tools to support the training process.

## **Folder Structure**

### **1. `config.yaml`**

-   This configuration file specifies the training parameters, dataset paths, and model-specific details.

### **2. `Train object_keypoint detection model.ipynb`**

-   A Jupyter Notebook that contains the code to train the object-keypoint detection model.
-   **How to Use**:
    -   Open this file in a Jupyter Notebook or Google Colab.
    -   Ensure the `config.yaml` file and datasets are correctly set up before running the training scripts.

### **3. `data/` Folder**

-   This folder contains the datasets required for training and validation.
-   **Contents**:
-   `images/`: Contains the images used for training and validation.
-   `labels/`: Contains the corresponding labels in the COCO keypoints format.

### **4. `CVAT to coco Keypoints/` Folder**

-   Contains a utility script for converting annotation files from **CVAT format** to **COCO Keypoints format**.
-   **File**: `CVAT_to_cocoKeypoints.py`
-   **Purpose**: If your dataset annotations are in CVAT format, use this script to convert them to COCO Keypoints format.
-   **Note**: If you already have labels in the correct COCO format, you can skip this script.
-   **Usage**: Run the script as follows:
    `python .\CVAT_to_cocoKeypoints.py`

---

## **Labels Format**

The labels used in this project follow the **COCO Keypoints format**, which includes bounding box coordinates and keypoints for each object in an image. Below is an example of the label format:

-   TODO: add a format example and explain the format.

---

## **How to Train the Model**

1. Ensure that your dataset (images and labels) is correctly set up in the `data/` folder.
2. Configure the paths and parameters in the `config.yaml` file.
3. Open and execute the `Train object_keypoint detection model.ipynb` notebook in your preferred environment (e.g., Google Colab).

-   The notebook will load the dataset, train the model, and save the resulting weights in the \*Current trained model\*\* folder.

---

# Steps for installing Ultralytics and Setting Up virtual Environment on Windows

## 1) Install Python

-   Download **Python 3.12** (version 3.13 may work, but revert to 3.12 if you encounter issues).
-   [Python Downloads Page](https://www.python.org/downloads/)

---

## 2) Set Up Python Virtual Environment

-   Navigate to your project folder.

-   Create a new Python virtual environment. Replace `env_netlist` with your preferred name if desired:
    ```bash
    py -m venv env_netlist
    ```

### Activate the Virtual Environment

To activate the virtual environment, run:

````bash
env_netlist\Scripts\activate

**Note:** You’ll know the environment is activated when you see the environment name in parentheses before the command prompt:

```lua
(env_netlist) C:\your\path\here>
````

## 3) Upgrade Pip

-   Run the following command to upgrade `pip`:
    ```bash
    pip install --upgrade pip
    ```

## 4) Install PyTorch

-   Go to the PyTorch website (https://pytorch.org/) and select:
    -   **Stable** version
    -   **Windows**
    -   **Pip**
    -   **Python**
    -   **CUDA 12.4** (for NVIDIA GPUs)
    -   Copy the provided install command and run it.
        Example command:
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```

## 5) Install Ultralytics

-   Install the `ultralytics` package (for YOLOv8):
    ```bash
    pip install ultralytics
    ```
