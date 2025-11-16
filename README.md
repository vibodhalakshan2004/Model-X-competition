# Dementia Risk Prediction Model

This project is a submission for the **[MODE-LX] Optimization Sprint**. The objective is to build a binary classification model to predict the risk of dementia using **non-medical features only**.

The model is designed to estimate dementia risk (0-100%) based on self-reported information that a person would know about themselves, such as demographics, lifestyle factors, and simple known health history.

## 📦 Dataset

The model is trained on the `Dementia Prediction Dataset.csv`.
Dataset Download Link : https://drive.google.com/file/d/1AtCCUDv8hGgmEUkUkMKNSKSO2qRoLmFE/view?usp=sharing

This is a longitudinal dataset containing **195,196 total visits** from **52,537 unique participants**. A key challenge and feature of this project is handling this data structure correctly.

To prevent data leakage, this pipeline uses a **`GroupShuffleSplit`** based on the participant ID (`NACCID`). This ensures that all visits from a single person are kept in the same data split (train, validation, or test), making the model's performance realistic and robust.

## 🚀 Installation

To set up this project locally, follow these steps:

1.  Clone this repository:
    ```sh
    git clone https://github.com/vibodhalakshan2004/Dementia-Prediction-Model-Notebook
    cd Dementia-Prediction-Model-Notebook
    ```
2.  Create and activate a virtual environment (recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## 💻 Usage

The entire analysis and model training pipeline is contained in the `Dementia Prediction Model Notebook.ipynb` file.

1.  Ensure you have the `Dementia Prediction Dataset.csv` in the root directory.
2.  Launch Jupyter Notebook or Jupyter Lab:
    ```sh
    jupyter lab
    ```
3.  Open `Dementia Prediction Model Notebook.ipynb` and run the cells from top to bottom.

## 🛠️ Project Workflow

The project follows a structured machine learning pipeline:

1.  **Data Loading:** The full dataset is loaded into a pandas DataFrame.
2.  **Feature Selection:** Allowed non-medical features are selected based on the Data Dictionary (primarily Forms A1, A5, and A3).
3.  **Feature Engineering:**
    * Raw data is cleaned (handling codes like `99`, `888`).
    * New features are created (e.g., `IS_MULTIRACIAL`, `IS_NON_ENGLISH_HOME`).
    * `BMI` is calculated from `HEIGHT` and `WEIGHT` to fill in missing values.
    * Categorical features (`MARISTAT`, `RESIDENC`, `RACE`) are one-hot encoded.
4.  **Data Splitting:** The data is split into train (70%), validation (15%), and test (15%) sets using `GroupShuffleSplit` on `NACCID`.
5.  **Model Training & Tuning:** Four different GBDT models are trained and tuned using `RandomizedSearchCV`:
    * Random Forest
    * XGBoost
    * CatBoost
    * LightGBM
6.  **Ensemble Modeling:** A **Stacking Ensemble** is created using a `LogisticRegression` meta-model to combine the predictions of the four base learners.
7.  **Evaluation:** All models are evaluated on the held-out test set using the **ROC-AUC** score.
8.  **Explainability:** The best-performing model (XGBoost) is analyzed with **SHAP** to understand which features are driving its predictions.

## 📊 Model Performance

The primary evaluation metric is **ROC-AUC**, which is ideal for imbalanced classification problems.

| Model | Validation AUC | Test AUC |
| :--- | :--- | :--- |
| Random Forest | 0.9344 | 0.9334 |
| XGBoost | 0.9369 | 0.9353 |
| CatBoost | 0.9377| 0.9367 |
| LightGBM | 0.9382 | 0.9362 |
| **Stacking Ensemble** | **0.9383** | **0.937** |

The **Stacking Ensemble** was selected as the final model due to its superior performance, effectively combining the strengths of all four base models.

## 🧠 Model Explainability (SHAP)

To understand what the model learned, a SHAP (SHapley Additive exPlanations) analysis was performed on the `XGBoost` model. The key insights are:

* **`AGE`:** This is the **most significant predictor** by a large margin. Higher age directly correlates with a much higher predicted risk of dementia.
* **`EDUC_YEARS`:** This is the second most important feature, acting as a **strong protective factor**. More years of education strongly decreased the predicted risk.
* **`INDEPEND`:** The participant's self-reported level of independence was a major factor. A lower level of independence (requiring assistance) was a strong indicator of risk.
* **Health History:** Self-reported conditions like `CBSTROKE_FLAG` (Stroke), `DIABETES_FLAG`, and `CVHATT_FLAG` (Heart Attack) were all significant risk-increasing features.
* **Social Factors:** `MARISTAT` (Marital Status) and `NACCLIVS` (Living Situation) also contributed, showing the model learned from the participant's social context.
