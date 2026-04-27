🦷 Multimodal Oral Cancer Detection using Deep Learning and Machine Learning

BSc (Hons) in Data Science and Business Analytics — Group Project (Year 3)  
 General Sir John Kotelawala Defence University

---

 📌 Overview

Oral cancer is one of the most prevalent cancers in Sri Lanka, and many cases are diagnosed at advanced stages, resulting in poor survival rates. Early detection is essential for improving treatment outcomes and reducing mortality.

This project develops a "multimodal oral cancer detection system" that integrates oral cavity images and patient metadata using both deep learning and machine learning techniques.

The system analyzes:
- 🖼️ Clinical oral cavity images — to detect visual abnormalities
- 📋 Patient risk factors — including smoking, betel chewing, alcohol consumption, age, and gender

By combining these two sources of information, the system aims to provide a more accurate and reliable oral cancer prediction model that can support early screening and clinical decision-making.

---

📂 Dataset

The dataset was collected through collaboration with Dental Science students from the University of Peradeniya and the Teaching Hospital Peradeniya. All patient data was collected with proper consent and used strictly for academic research purposes.

Dataset Components

🖼️ Oral Image Dataset
- Approximately 3,000 oral cavity images
- Captured under clinical conditions
- Covers multiple oral health conditions

📋 Patient Metadata
- 714 patient records
- Each record includes key risk factors:
  - Age
  - Gender
  - Smoking habit
  - Betel chewing habit
  - Alcohol consumption

Dataset Classes

The original dataset contains four medical categories:


OCA - Oral Cancer 
OPMD - Oral Potentially Malignant Disorders 
 Benign -  Non-cancerous lesions 
 Healthy - Normal oral cavity 

For binary classification, classes were grouped as follows:


✅ Cancerous -  OCA + OPMD 
 ❌ Non-Cancerous - Benign + Healthy 



🎯 Project Objectives

1. Develop deep learning models to detect oral cancer from oral cavity images.
2. Develop machine learning models to analyze patient metadata and risk factors.
3. Combine image-based and metadata-based predictions using a multimodal learning approach.
4. Compare the performance of single modality vs multimodal models.
5. Provide a system that can assist in early oral cancer detection.



🔬 Methodology

1. Data Collection
Data was obtained through collaboration with University of Peradeniya Dental Science students and Teaching Hospital Peradeniya, comprising both clinical images and patient metadata.

 2. Data Preprocessing

 Images - Resizing and normalization 
 Tabular Data - Cleaning, encoding categorical variables, feature scaling 
 Missing Values -  Handled appropriately 
Class Imbalance - Balanced class distribution applied 

3. Exploratory Data Analysis (EDA)
EDA was conducted to understand relationships between risk factors and oral cancer, including:
- Age and gender distribution
- Smoking, betel chewing, and alcohol consumption patterns

Statistical tests used: Chi-Square Test, t-test

 4. Model Development

🧠 Image-Based Models (Deep Learning)
Deep learning models were used to detect patterns in oral cavity images.


 Convolutional Neural Networks (CNN) -  Feature extraction from images 
Transfer Learning - Leverage pre-trained models 
Vision Transformers - Attention-based image understanding 
Data Augmentation - Improve generalization 

📊 Metadata-Based Models (Machine Learning)
Machine learning models were trained using patient risk factors.


Logistic Regression - Baseline model 
 Random Forest - Ensemble method 
K-Nearest Neighbors - Distance-based 
Support Vector Machine - Margin-based classifier 
Gradient Boosting- Sequential ensemble 
XGBoost - Optimized boosting 
LightGBM -⭐ Best performing model 
CatBoost - Handles categorical data 

> LightGBM achieved the best performance for metadata classification.

 5. Multimodal Fusion

The project uses a late fusion approach:

```
Image Model → Prediction ──┐
                            ├──► Final Classification
Metadata Model → Prediction ┘
```

This improves prediction reliability by combining visual features with patient information.

 6. Dataset Split


Training Set - 70% 
Validation Set - 10% 
Testing Set - 20% 

 7. Model Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

🚀 How to Run

 Prerequisites
Install all required dependencies:

```bash
pip install -r requirements.txt
```

Run Notebooks Step by Step

```bash
jupyter notebook
```

Open and run the notebooks in the following order:

1️⃣  Data Preprocessing 
2️⃣  Exploratory Data Analysis 
3️⃣  Train Image Models 
4️⃣  Train Metadata Models 
5️⃣  Multimodal Fusion Model 

---

🛠️ Dependencies

```txt
numpy
pandas
scikit-learn
pytorch
torchvision
timm
matplotlib
seaborn
xgboost
lightgbm
catboost
```

Install using:

```bash
pip install -r requirements.txt
```

---

⚠️ Data Confidentiality Notice

> The dataset used in this project **cannot be publicly shared.

The oral cavity images and patient metadata were collected through the University of Peradeniya and Teaching Hospital Peradeniya under ethical research considerations. Due to **medical data confidentiality and patient privacy**, the dataset cannot be uploaded to this repository.

Researchers interested in replicating this study must obtain approval from the relevant medical institutions.

---

💡 Expected Impact

This project demonstrates how artificial intelligence can assist healthcare professionals in early cancer detection.

⚡ Faster Screening - Reduces time to identify oral cancer cases 
🔍 Early Diagnosis - Flags high-risk patients at an earlier stage 
 💊 Better Outcomes - Improved patient survival rates through early intervention 
 🏥 Healthcare Support  - Assists systems with limited specialist availability 

---

👥 Authors

Group Project — BSc (Hons) in Data Science and Business Analytics

 KAS Samadine 
JAC Sudarshika  
 WDK Shihara 
 DMDN Dissanayake 

Supervisor: Dr. Chithraka Wickramarachchi

---

 📄 License

This project is for academic purposes only. All rights reserved by the respective authors and institutions.


