# Credit Access Risk Prediction - Flask Deployment on AWS

This project is designed to predict credit risk using machine learning. It utilizes a Flask API for model deployment and is hosted on AWS, providing real-time predictions for financial institutions to make informed decisions.

---

## Features
- **Machine Learning Model**: Predicts credit risk with high accuracy using trained algorithms.
- **Flask Integration**: API for model interaction and seamless integration with other systems.
- **AWS Deployment**: Deployed on AWS to ensure scalability and availability.
- **Automation**: Integrated database validation scripts and efficient pipeline management with GitHub Actions.

---

## Tech Stack
- **Programming Language**: Python
- **Framework**: Flask
- **Machine Learning**: Scikit-learn
- **Deployment Platform**: AWS Elastic Beanstalk
- **Data Handling**: Pandas, NumPy
- **Automation**: GitHub Actions

---

## Folder Structure
```plaintext
credit_access_flask_deployment/
├── static/            # Static files (if any)
├── templates/         # HTML templates for Flask (if any)
├── app.py             # Main application script
├── model.pkl          # Serialized machine learning model
├── requirements.txt   # Python dependencies
├── Procfile           # Elastic Beanstalk configuration
├── .ebextensions/     # AWS Elastic Beanstalk config files
└── README.md          # Documentation
```

## Installation and Usage

### Prerequisites
Before running the application, ensure you have the following installed:
1. **Python** (Version 3.7 or higher)
2. **Pip** (Python's package manager)
3. **AWS Elastic Beanstalk CLI** (For deployment)

### Steps to Run Locally
1. **Clone the Repository**  
   Clone the project to your local machine:
   ```bash
   git clone https://github.com/kvamsi7/ML-portfolio.git
   cd "Acess Credit risk - AWS deployment/credit_access_flask_deployment"

