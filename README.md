Document Cleaning Pipeline
A machine-learning-powered pipeline for cleaning scanned documents. This project leverages a pre-trained Denoising Convolutional Neural Network (DnCNN) combined with rule-based post-processing techniques to remove noise, enhance text clarity, and produce clean, professional-looking documents.

Features
Denoising with CNN:
Uses a pre-trained DnCNN model to remove background noise and artifacts.
Adaptive Thresholding:
Binarizes the image for uniform background and sharper text.
Morphological Operations:
Removes small artifacts and smoothens the text for enhanced readability.
Customizable Pipeline:
Easily adjustable to include additional processing steps (e.g., segmentation, batch processing).
Requirements
To run the project, install the following dependencies:

bash
Copy
Edit
pip install torch torchvision opencv-python numpy
Usage
Clone the repository:

bash
Copy
Edit
git clone https://github.com/<your-username>/document-cleaning-pipeline.git
cd document-cleaning-pipeline
Run the pipeline:

bash
Copy
Edit
python document_cleaning_pipeline.py
The cleaned document will be saved as final_document.png.

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
└── README.md  # Project documentation
Example
Input (Noisy Document):

Output (Cleaned Document):

Next Steps
Add batch processing to clean multiple documents at once.
Implement PDF generation from cleaned images.
Fine-tune the DnCNN model on your specific dataset for better results.
Contributing
Contributions are welcome! Feel free to fork this repository and submit a pull request with your improvements.

License
This project is licensed under the MIT License. See the LICENSE file for details.

