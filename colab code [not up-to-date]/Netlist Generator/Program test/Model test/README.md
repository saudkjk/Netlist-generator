# **Model Test**

The **Model Test** folder provides the tools and resources required to test the performance of the trained object-keypoint detection model. This folder allows users to evaluate how well the model performs on sample circuit schematic images by visualizing bounding boxes and keypoints.

## **Folder Structure**

### **1. Test Images**
- A folder containing sample images for testing the trained model.

### **2. `Model Test.ipynb`**
- A Jupyter Notebook for testing the trained model on the images provided in the `Test Images` folder.
- **Functionality**:
  - Processes all images in the `Test Images` folder.
  - Outputs visualized results:
    - **Blue Boxes**: Drawn around detected components in the schematic.
    - **Green Dots**: Marked at the keypoints representing the connection points of each component.

---

## **How to Use**

1. Place your test images in the `Test Images` folder.
2. Open the `Model Test.ipynb` file in Jupyter Notebook or Google Colab.
3. Run the notebook to:
   - Process all the images in the `Test Images` folder.
   - View the model's output visualizations, including bounding boxes and keypoints, directly in the notebook.

---
