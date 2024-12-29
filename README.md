# **Netlist Generator**

Welcome to the **Netlist Generator** repository! This project is designed to convert an image of a circuit schematic into a netlist, which is outputted as a text file. The netlist represents the electrical connections within the circuit and is a crucial tool in circuit design and simulation.

## **Folder Structure**

### **1. Current Trained Model**
- Contains the latest trained machine learning model used to analyze and interpret the circuit schematic images.
- Includes:
  - The model weights and training arguments.
  - Old models for testing purposes.
  - Documentation of the model's performance metrics.

### **2. Model Training**
- Holds all the resources needed to train the machine learning model.
- Includes:
  - Datasets used for training.
  - Preprocessing scripts for preparing the data.
  - Training scripts and configuration files.
  - Information on the training pipeline and how to retrain or fine-tune the model.
  - Tool to convert CVAT format to coco Keypoints format.

### **3. Program Test**
- Provides tools and scripts to test the functionality of the trained model and the overall program.
- Includes:
  - Script to test the model performance.
  - Script to test netlists generation methods.
  - Example netlist outputs for reference.
  - Example image outputs for debugging.
  - Script to compare different netlist generation methods.
  - Correctly constructed netlist files for testing.

---

## **How to Use This Repository**

1. **Getting Started**: Begin by reading this root-level README to understand the overall purpose and structure of the project.
2. **Exploring Folders**: Navigate to individual folders for more detailed instructions and resources.
3. **Running the Program**:
   - Use the `Program Test` folder to test the functionality.
   - Follow the instructions provided in the folder's README to run the program and generate netlists.

---
