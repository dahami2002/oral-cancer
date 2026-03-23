Multimodal Oral Cancer Detection using Deep Learning and Machine Learning
BSc (Hons) in Data Science and Business Analytics — Group Project (Year 3)
General Sir John Kotelawala Defence University
Overview
Oral cancer is one of the most prevalent cancers in Sri Lanka, and many cases are diagnosed at advanced stages, resulting in poor survival rates. Early detection is essential for improving treatment outcomes and reducing mortality.
This project develops a multimodal oral cancer detection system that integrates oral cavity images and patient metadata using both deep learning and machine learning techniques.
The system analyzes:
•	Clinical oral cavity images to detect visual abnormalities
•	Patient risk factors such as smoking, betel chewing, alcohol consumption, age, and gender
By combining these two sources of information, the system aims to provide a more accurate and reliable oral cancer prediction model that can support early screening and clinical decision-making.

Dataset
The dataset used in this project was collected through collaboration with Dental Science students from the University of Peradeniya and the Teaching Hospital Peradeniya.
All patient data was collected with proper consent and used strictly for academic research purposes.
Dataset Components
The dataset consists of two main parts:
1. Oral Image Dataset
•	Approximately 3,000 oral cavity images
•	Images captured under clinical conditions
•	Different oral health conditions included
2. Patient Metadata
•	714 patient records
•	Each record contains important risk factors such as:
•	Age
•	Gender
•	Smoking habit
•	Betel chewing habit
•	Alcohol consumption

Dataset Classes
The original dataset contains four medical categories:
•	OCA – Oral Cancer
•	OPMD – Oral Potentially Malignant Disorders
•	Benign – Non-cancerous lesions
•	Healthy – Normal oral cavity
For the binary classification task, the classes were grouped as follows:
Group	Classes
Cancerous	OCA + OPMD
Non-Cancerous	Benign + Healthy

Project Objectives
The main objectives of this project are:
1.	Develop deep learning models to detect oral cancer from oral cavity images.
2.	Develop machine learning models to analyze patient metadata and risk factors.
3.	Combine image-based and metadata-based predictions using a multimodal learning approach.
4.	Compare the performance of single modality models vs multimodal models.
5.	Provide a system that can assist in early oral cancer detection.

Methodology
The project follows several stages from data preprocessing to model evaluation.
1. Data Collection
The data was obtained through collaboration with University of Peradeniya Dental Science students and Teaching Hospital Peradeniya. The dataset contains both clinical images and patient metadata.

2. Data Preprocessing
To ensure data quality, the following preprocessing steps were applied:
•	Image resizing and normalization
•	Data cleaning
•	Encoding categorical variables
•	Feature scaling
•	Handling missing values
•	Balancing class distribution

3. Exploratory Data Analysis (EDA)
EDA was conducted to understand relationships between risk factors and oral cancer.
Examples include:
•	Age distribution
•	Gender distribution
•	Smoking habits
•	Betel chewing habits
•	Alcohol consumption patterns
Statistical tests used:
•	Chi-Square Test
•	t-test

4. Model Development
Two types of models were developed.

Image-Based Models (Deep Learning)
Deep learning models were used to detect patterns in oral cavity images.
Techniques used:
•	Convolutional Neural Networks (CNN)
•	Transfer Learning
•	Vision Transformer Models
•	Data Augmentation
These models automatically learn features related to oral lesions and abnormalities.

Metadata-Based Models (Machine Learning)
Machine learning models were trained using patient risk factors.
Algorithms used:
•	Logistic Regression
•	Random Forest
•	K-Nearest Neighbors
•	Support Vector Machine
•	Gradient Boosting
•	XGBoost
•	LightGBM
•	CatBoost
Among these, LightGBM achieved the best performance for metadata classification.

Multimodal Fusion
The project uses a late fusion approach.
Steps:
1.	Image model generates predictions
2.	Metadata model generates predictions
3.	Predictions are combined to generate the final classification result
This improves prediction reliability by using both visual features and patient information.

Model Training
The dataset was split as follows:
Dataset	Percentage
Training Set	70%
Validation Set	10%
Testing Set	20%
Training techniques included:
•	Transfer learning
Model Evaluation
The models were evaluated using the following metrics:
•	Accuracy
•	Precision
•	Recall
•	F1-Score
•	Confusion Matrix
•	Classification Report
These metrics help evaluate the effectiveness of the model in identifying cancerous and non-cancerous cases.
How to Run the Project
Run the notebooks step by step:
1️⃣ Data preprocessing
2️⃣ Exploratory data analysis
3️⃣ Train image models
4️⃣ Train metadata models
5️⃣ Run multimodal fusion model
Example:
jupyter notebook
Open the notebooks and run each cell sequentially.
________________________________________
Dependencies
The following libraries were used in this project.
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
Install them using:
pip install -r requirements.txt

Data Confidentiality Notice
⚠️ The dataset used in this project cannot be publicly shared.
The oral cavity images and patient metadata were collected through University of Peradeniya and Teaching Hospital Peradeniya under ethical research considerations.
Due to medical data confidentiality and patient privacy, the dataset cannot be uploaded to this repository.
Researchers interested in replicating this study must obtain approval from the relevant medical institutions.

Expected Impact
This project demonstrates how artificial intelligence can assist healthcare professionals in early cancer detection.
Potential benefits include:
•	Faster screening of oral cancer cases
•	Early diagnosis support
•	Improved patient survival rates
•	Support for healthcare systems with limited specialists

Authors
Group Project — BSc (Hons) in Data Science and Business Analytics
KAS Samadine
JAC Sudarshika
WDK Shihara
DMDN DissanayakeSupervisor(s):
•	Project Supervisor Name :  Dr.Chithraka Wickramarachchi
.

