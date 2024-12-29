# **Model Training**

This folder contains all the resources needed to train the machine learning model used in the **Netlist Generator** project. It includes datasets, scripts, configuration files, and tools to support the training process.

## **Folder Structure**

### **1. `config.yaml`**
- This configuration file specifies the training parameters, dataset paths, and model-specific details.
  
### **2. `Train object_keypoint detection model.ipynb`**
- A Jupyter Notebook that contains the code to train the object-keypoint detection model.
- **How to Use**:
  - Open this file in a Jupyter Notebook or Google Colab.
  - Ensure the `config.yaml` file and datasets are correctly set up before running the training scripts.

### **3. `data/` Folder**
- This folder contains the datasets required for training and validation.
- **Contents**:
- `images/`: Contains the images used for training and validation.
- `labels/`: Contains the corresponding labels in the COCO keypoints format.

### **4. `CVAT to coco Keypoints/` Folder**
- Contains a utility script for converting annotation files from **CVAT format** to **COCO Keypoints format**.
- **File**: `CVAT_to_cocoKeypoints.py`
- **Purpose**: If your dataset annotations are in CVAT format, use this script to convert them to COCO Keypoints format.
- **Note**: If you already have labels in the correct COCO format, you can skip this script.
- **Usage**: Run the script as follows:
```python .\CVAT_to_cocoKeypoints.py```

---

## **How to Train the Model**

1. Ensure that your dataset (images and labels) is correctly set up in the `data/` folder.
2. Configure the paths and parameters in the `config.yaml` file.
3. Open and execute the `Train object_keypoint detection model.ipynb` notebook in your preferred environment (e.g., Google Colab).
 - The notebook will load the dataset, train the model, and save the resulting weights in the *Current trained model** folder.

---

## **Labels Format**

The labels used in this project follow the **COCO Keypoints format**, which includes bounding box coordinates and keypoints for each object in an image. Below is an example of the label format:
- TODO: add a format example and explain the format.

---

## **Dependencies**

To run CVAT_to_cocoKeypoints.py, ensure you have the following:
- **Python** (version 3.7 or higher)
- Required libraries (install via `pip install -r requirements.txt` if available):
- TODO: Add requirements.txt

---

