# 🎬 Movie Genre Classification

## 📂 Input
- A **CSV file** with at least two columns:
  - `plot` → text description of the movie (or any chosen text column)  
  - `genre` → target label (the genre of the movie)

---

## 🚀 Steps Performed in the Code

### 1. Load Dataset
- Read the CSV file using **pandas**.
- Validate that the required columns (`plot`, `genre`) exist.
- Convert text to string and replace null values with an empty string.
- Split into **train (80%)** and **test (20%)** using `train_test_split` with stratification.

**What I learned**: Checking for required columns prevents runtime errors. Stratified split ensures class balance.

---

### 2. Build Pipelines
- Use **TF-IDF Vectorizer** with:
  - Lowercasing text  
  - English stopword removal  
  - N-grams (unigrams + bigrams)  
  - Minimum document frequency = 2  
  - Max features = 100,000  
- Create ML pipelines combining TF-IDF with three classifiers:
  - **Multinomial Naive Bayes**
  - **Logistic Regression**
  - **Linear Support Vector Classifier**

**What I learned**: Pipelines simplify preprocessing + modeling in one object. TF-IDF converts text to numerical vectors effectively.

---

### 3. Cross-Validation (Model Selection)
- Perform **5-fold Stratified Cross-Validation** on the training data.
- Metric: **Macro F1-score** (treats all classes equally).
- Track mean and standard deviation of F1 for each model.
- Select the **best-performing model** based on mean macro F1.

**What I learned**:  
- Cross-validation reduces overfitting.  
- Macro F1 is better than accuracy when classes are imbalanced.  
- Automating model selection ensures fairness across algorithms.  

---

### 4. Train Best Model
- Fit the selected model on the **entire training set**.
- Make predictions on the **test set**.

**What I learned**: Always finalize training on the full train split before testing.

---

### 5. Evaluate Model
- Print **classification report** (precision, recall, F1-score per class).  
- Print **confusion matrix** to visualize predictions vs. true labels.  

**What I learned**:  
- Classification report gives a per-class view.  
- Confusion matrix helps spot where models confuse genres.  

---

### 6. Save Model & Metadata
- Save trained pipeline and metadata (best model name, CV results, column names, test size) using **joblib**.
- Default output file: `movie_genre_model.joblib`.

**What I learned**: Saving the pipeline allows reusing the model later without retraining. Metadata makes the model self-documented.

---
