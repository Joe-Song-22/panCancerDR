![image](https://github.com/user-attachments/assets/9e5f99f0-2eeb-4e12-bcb9-cb36c3801908)

panCancerDR: A Domain Generalization Framework for Drug Response Prediction
panCancerDR is a novel domain generalization framework designed to predict drug response in out-of-distribution samples, including individual cells and patient data, using only in vitro cancer cell line data for training. By leveraging adversarial domain generalization and innovative feature extraction techniques, panCancerDR addresses the limitations of traditional domain adaptation methods, which are unsuitable for unseen target domains.
Key Features
Latent Independent Projection (LIP): A plug-and-play module that extracts expressive and decorrelated features to mitigate overfitting and improve generalization across diverse datasets.
Asymmetric Adaptive Clustering Constraint: Ensures sensitive samples are tightly clustered, while resistant samples form diverse clusters in the latent space, reflecting real-world biological diversity in drug responses.
Comprehensive Validation: The model has been rigorously evaluated on bulk RNA-seq, single-cell RNA-seq (scRNA-seq), and patient-level datasets, demonstrating superior or comparable performance to current state-of-the-art (SOTA) methods.
Contributions
Novel LIP Module: Encourages the encoder to extract informative, non-redundant features, enabling robust predictions across unseen domains.
Asymmetric Clustering for Generalization: Inspired by the distinct characteristics of sensitive and resistant cells, this approach ensures effective latent space representation for drug response prediction.
Extensive Evaluation: Achieved high performance on 13 cancer types, single-cell data, and patient-level drug response tasks, highlighting the model's versatility and predictive power.
Here’s a comprehensive README file for your project:

---

# **Drug-Specific Data Analysis and Modeling**

This repository contains code and resources for analyzing drug-specific data and building machine learning models using PyTorch. The project leverages advanced GPU computation (NVIDIA 4090 GPU) to efficiently process large datasets and train complex neural network architectures.

## **Project Overview**

The primary goal of this project is to investigate drug-specific effects and build predictive models based on corresponding datasets. Each dataset corresponds to a specific drug and is stored in a separate file.

## **Prerequisites**

Before running the code, ensure you have the following libraries installed:

- **Python Libraries**:
  - `pandas`: For data manipulation and preprocessing.
  - `torch`: PyTorch framework for building and training models.
  - `torchvision`: For additional PyTorch utilities (if required).
  - `sklearn`: For data splitting, metrics, and other utilities.
  - `tqdm`: For progress bar visualization.
  - `numpy`: For numerical operations.
  - `matplotlib`: For plotting and visualizing results.
  
- **Hardware Requirements**:
  - **GPU**: NVIDIA 4090 or equivalent is required for training large-scale models.

## **Setup**

### **1. Clone the Repository**
```bash
git clone <repository_url>
cd <repository_name>
```

### **2. Install Dependencies**
Use the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

### **3. GPU Configuration**
Ensure your system has an NVIDIA GPU with the appropriate drivers and CUDA toolkit installed. Verify PyTorch detects the GPU:
```python
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### **4. File Organization**
Each file corresponds to a drug-specific dataset. Ensure the data files are correctly placed in the root directory. Example structure:
```
root/
├── drug1_data.csv
├── drug2_data.csv
├── model_test.py
├── main.py
├── utils.py
```

## **Usage**

### **1. Data Preparation**
- Each dataset is preprocessed using `pandas` and split into training and testing sets using `train_test_split`.
- Shuffle your data using `sklearn.utils.shuffle` for better generalization.

### **2. Model Training**
- Custom models are defined in the `model_test.py` file.
- Load the data, preprocess it, and train the model as follows:
```python
from model_test import MyModel
from torch.utils.data import DataLoader, TensorDataset

# Load and preprocess data
data = pd.read_csv('drug1_data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split and shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize and train model
model = MyModel().to(device)
```

### **3. Evaluation and Visualization**
- Evaluate the model's performance using metrics such as AUC and visualize results with matplotlib:
```python
from sklearn.metrics import roc_auc_score

y_true = ...
y_pred = ...
auc = roc_auc_score(y_true, y_pred)
print(f"AUC: {auc}")
```

### **4. Plot ROC Curve**
```python
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_true, y_pred)
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

## **Project Highlights**
- Efficient data processing using `pandas` and `sklearn`.
- Scalable and customizable PyTorch models in `model_test.py`.
- Utilization of NVIDIA 4090 GPU for high-performance model training.
- Comprehensive metrics and visualization for model evaluation.

## **Contributing**
Feel free to open issues or submit pull requests to improve the project. Your contributions are welcome!

## **License**
This project is licensed under the MIT License. See the LICENSE file for more details.

---

This README provides all necessary information to understand, set up, and run your project. Let me know if you'd like any additional sections or modifications!
# panCancerDR
