# **Current Trained Model**

This folder contains the resources related to the machine learning models used for analyzing and interpreting circuit schematic images. The models are designed to detect circuit components and their connections to generate a netlist. The folder is structured into two subfolders:

## **Folder Structure**

### **1. Old Models for Testing Purposes**
- This folder contains older versions of the trained models.
- These models are kept for testing, experimentation, or reference purposes.
- Includes:
  - Legacy models from previous training iterations.
  - Details on their training configuration and performance metrics.

### **2. Pose Folder**
- This folder contains the **current trained model**, which is the result of training using **Ultralytics YOLO** on Google Colab.
- The model is a **pose detection version of YOLO** and has been adapted for this project to meet specific requirements:
  - **Bounding Box Detection**: Identifies the circuit components in the schematic.
  - **Keypoint Detection**: Extracts keypoints that represent the connection points of each component to the rest of the circuit.

---

## **Why Pose Detection?**

The use of a pose detection model was chosen for its ability to handle two critical tasks simultaneously:
1. **Bounding Box Detection**: Localizing circuit components in the schematic image.
2. **Keypoint Detection**: Identifying the connection points (pins) of the components, which is essential for constructing the netlist.

By combining these two features, the model provides detailed information about each component and its connections, enabling the generation of an accurate netlist.

---

## **How to Use This Folder**

1. **Using the Current Model**:
   - Pass the following model path to other programs that use the model for inference:
     ```python
     model_path = '/content/gdrive/My Drive/Netlist generator/Current trained model/pose/train/weights/last.pt'
     ```
   - Ensure that your program is configured to load the model correctly from this path.
   - Replace `/content/gdrive/My Drive/` with the appropriate base path if the directory structure differs.

2. **Testing Older Models**:
   - Navigate to the **Old Models for Testing Purposes** folder to access older versions of the model.
   - You can use these models for comparison or testing purposes.

---
