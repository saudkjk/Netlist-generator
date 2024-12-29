# **Program Test**

The **Program Test** folder provides tools and scripts to test the functionality of the trained model and the overall netlist generation process. It serves as the validation and debugging hub for the project, ensuring the model and netlist generation algorithms perform as expected.

## **Folder Structure**

### **1. Model Test**
- This folder contains resources and scripts to test the performance of the trained object-keypoint detection model.
- Includes:
  - Scripts for evaluating the model's performance on testing datasets.
  - Example input images to test model inference.
  - Outputs such as bounding boxes, keypoints, and other detection results for debugging purposes.

### **2. Netlist Generation Algorithm Test**
- This folder contains resources to test and validate the methods used for generating netlists from the outputs of the trained model.
- Includes:
  - Scripts for generating netlists from the model's outputs.
  - Example netlists produced during testing.
  - Scripts to compare different netlist generation methods.
  - Correctly constructed netlist files for use as ground truth during testing and debugging.

---

## **Purpose of This Folder**

1. **Testing the Model**:
   - Ensure the trained object-keypoint detection model performs as expected on testing datasets.
   - Debug issues such as incorrect bounding boxes or keypoints.

2. **Validating Netlist Generation**:
   - Verify that the netlist generation algorithms produce correct and consistent netlists.
   - Compare different methods for generating netlists and determine the most effective approach.
   - Use correctly constructed netlist files as references to evaluate the quality of generated netlists.

---
