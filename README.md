# ML Model CI/CD Pipeline

This project demonstrates a CI/CD pipeline for a simple CNN model trained on MNIST dataset.

## Project Structure 

── .github/
│ └── workflows/
│ └── ml_pipeline.yml
├── src/
│ ├── train.py
│ ├── model.py
│ └── test_model.py
├── requirements.txt
└── README.md

## Features
- Convolutional Neural Network for MNIST classification
- Automated testing and validation
- GitHub Actions CI/CD pipeline
- Model parameter count < 25,000
- Target accuracy > 95% in first epoch

## Local Setup

1. Create a virtual environment:bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
2. Install dependencies:
bash
pip install -r requirements.txt
3. Run training locally:
bash
python src/train.py
4. Run tests:
bash
pytest src/test_model.py

## GitHub Actions
The pipeline automatically:
1. Sets up Python environment
2. Installs dependencies
3. Runs model training
4. Executes tests
5. Validates model requirements
6. Creates a model artifact with timestamp

## Model Architecture
- Input Layer: 28x28 images
- Conv2D Layer: 8 filters
- Conv2D Layer: 16 filters
- Fully Connected Layer: 10 outputs
- Total parameters: < 25,000

This setup includes:
A model architecture with < 25,000 parameters
Training script with assertions for accuracy and parameter count
3. Test suite checking:
Model architecture
Input/output shapes
Parameter count
Prediction format
Training functionality
GitHub Actions pipeline that:
Sets up the environment
Runs tests
Trains model
Uploads the trained model as an artifact