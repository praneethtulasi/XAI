
# XAI-Based Chart Detection, Classification, and Insight Generation

This project presents an Explainable AI (XAI) system designed to automatically detect, classify, and extract insights from charts and graphs embedded in PDF or image-based documents. The system integrates deep learning and interpretability methods to enable transparent and trustworthy analysis of visual data.

## Project Objectives

- Automatically detect chart regions within documents.
- Classify the type of chart (e.g., bar, pie, line, scatter).
- Generate meaningful insights from the chart data using natural language.
- Provide visual explanations of model predictions to support interpretability.

## Features

- **Chart Detection**: Utilizes YOLOv8 to locate and isolate charts from input documents.
- **Chart Classification**: Differentiates among various chart types with high accuracy.
- **Insight Generation**: Employs models such as ChartQA, Gradient, and LLaMA to produce textual insights and answer questions about the charts.
- **Explainability**: Applies Grad-CAM and related techniques to visualize model focus areas and support decision transparency.

## Technology Stack

- Python (core development)
- YOLOv8 (chart detection and classification)
- ChartQA, Gradient, LLaMA (insight generation)
- Grad-CAM (model explainability)
- Streamlit (optional interface for interactive analysis)
- OpenCV and Matplotlib (for preprocessing and visualization)

## Use Cases

- Automated chart analysis in research papers and reports
- Business intelligence and reporting tools
- Accessible data interpretation for non-technical users
- Visual data auditing and compliance workflows

## Folder Structure

```

├── data/                   # Sample charts or PDFs
├── models/                 # Pre-trained model files
├── utils/                  # Helper functions and preprocessing scripts
├── app.py                  # Main application (Streamlit or Flask)
├── requirements.txt        # Required Python packages
├── README.md               # Project documentation

````

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/praneethtulasi/XAI.git
   cd XAI
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application (if using Streamlit):

   ```bash
   streamlit run app.py
   ```

## Future Improvements

* Integrate tabular data extraction from charts
* Improve multilingual insight generation
* Incorporate user feedback mechanisms for explainability validation
* Extend support to complex and 3D visualizations

## License

This project is released under the MIT License.

