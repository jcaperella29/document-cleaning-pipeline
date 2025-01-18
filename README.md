# Document Cleaning Pipeline 🧹📝

This repository contains a Python-based pipeline for cleaning scanned document images. The pipeline leverages a **DnCNN-based convolutional neural network** for denoising, coupled with adaptive thresholding and post-processing, to generate clean, readable outputs that are ideal for both **human readability** and **text mining**.

---

## **Features** ✨

- **Denoising with DnCNN**:
  Uses a pre-trained DnCNN (Deep Convolutional Neural Network) to remove noise while preserving important text details.
  
- **Adaptive Thresholding**:
  Sharpens text, enhances contrast, and creates uniform backgrounds for better readability and machine processing.

- **PDF Conversion**:
  Converts cleaned images into grayscale, high-resolution PDFs for archival and text mining.

- **Batch Processing**:
  Processes all images in a folder and generates cleaned images and PDFs in bulk.

---

## **Example Output**

### **Input (Noisy Image)**
![Noisy Input](example_images/noisy.png)

### **Output (Cleaned Image)**
![Cleaned Output](example_images/cleaned.png)

### **Output (Thresholded Binary)**
![Thresholded Output](example_images/thresholded.png)

---

## **Installation** 🛠️

### 1. **Clone the Repository**
```bash
git clone https://github.com/jcaperella29/document-cleaning-pipeline.git
cd document-cleaning-pipeline
