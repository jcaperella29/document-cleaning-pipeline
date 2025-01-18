Document Cleaning Pipeline
A machine-learning-powered pipeline for cleaning scanned documents. This project leverages a pre-trained Denoising Convolutional Neural Network (DnCNN) combined with rule-based post-processing techniques to remove noise, enhance text clarity, and produce clean, professional-looking documents. The pipeline also supports exporting cleaned images as PDFs.

Features
Denoising with CNN:
Uses a pre-trained DnCNN model to remove background noise and artifacts.
Adaptive Thresholding:
Binarizes the image for uniform background and sharper text.
Morphological Operations:
Removes small artifacts and smoothens the text for enhanced readability.
PDF Conversion:
Exports the cleaned document as a high-quality PDF.
Requirements
To run the project, install the following dependencies:

Copy
Edit
pip install torch torchvision opencv-python numpy pillow
Usage
Clone the repository:

bash
Copy
Edit
git clone https://github.com/jcaperella29/document-cleaning-pipeline.git
cd document-cleaning-pipeline
Run the pipeline:

Copy
Edit
python document_cleaning_pipeline.py
Outputs:

Cleaned Image: Saved as final_document.png.
PDF File: Saved as final_document.pdf.
Directory Structure
bash
Copy
Edit
document-cleaning-pipeline/
├── document_cleaning_pipeline.py  # Main script
├── models/
│   └── DnCNN_sigma25/
│       └── model.pth  # Pre-trained weights
├── data/
│   └── val/
│       └── noisy_0.png  # Example noisy image
├── README.md  # Project documentation
Example
Input (Noisy Document):
An example noisy document containing smudges and artifacts.

Output (Cleaned Document):
The cleaned version with enhanced text clarity and uniform background.

Next Steps
Add batch processing to clean multiple documents at once.
Implement multi-page PDF generation from multiple cleaned images.
Build a simple GUI for easier usage.
Contributing
Contributions are welcome! Feel free to fork this repository and submit a pull request with your improvements.

License
This project is licensed under the MIT License. See the LICENSE file for details.
