# Spambase Email Threat Classification Suite

## a. Problem Statement
Unsolicited commercial and malicious bulk emails (spam) pose persistent operational and security risks, ranging from inbox clutter to phishing attacks. The objective of this project is to build, evaluate, and deploy an end-to-end supervised binary classification pipeline capable of distinguishing **Spam (`1`)** from **Legitimate (`0`)** emails based on quantitative lexical and character-level frequency signatures. Five distinct machine learning classifiers are trained, evaluated across six comprehensive performance metrics, and deployed via an interactive Streamlit web application for real-time test set inference.

---

## b. Dataset Description
* **Dataset Name:** Spambase Dataset
* **Source Repository:** UCI Machine Learning Repository (also mirrored on Kaggle / OpenML)
* **Problem Type:** Binary Classification (`0` = Not Spam / Legitimate, `1` = Spam)
* **Total Instances:** 4,601 emails (1,813 Spam [39.4%] and 2,788 Legitimate [60.6%]) — *exceeds the minimum requirement of 500 instances.*
* **Total Input Features:** 57 continuous numeric features — *exceeds the minimum requirement of 12 features:*
  * **48 Word Frequency Features (`word_freq_*`):** Percentage (`[0, 100]`) of words in the email matching specific keywords (e.g., `free`, `remove`, `money`, `credit`, `hp`, `george`, `meeting`).
  * **6 Character Frequency Features (`char_freq_*`):** Percentage (`[0, 100]`) of characters in the email matching `;`, `(`, `[`, `!`, `$`, and `#`.
  * **3 Capital Letter Run-Length Features (`capital_run_length_*`):** Average length of uninterrupted sequences of capital letters (`average`), length of the longest sequence (`longest`), and total number of capital letters in the email (`total`).
* **Preprocessing & Data Split:** Stratified train-test split (`78%` training = 3,588 instances; `22%` testing = 1,013 instances, exported as `test_data.csv` with `random_state=104`). Feature standardization (`StandardScaler`) is encapsulated directly inside each `scikit-learn` `Pipeline` artifact to prevent data leakage and allow seamless test inference without retraining.

---

## c. Github Repository Link
* **GitHub Repository Link:** [https://github.com/balrajprajesh/ml-assignment-2](https://github.com/balrajprajesh/ml-assignment-2)
* **Live Streamlit App Link:** [https://ml-assignment-2-2026aj05123.streamlit.app/](https://ml-assignment-2-2026aj05123.streamlit.app/)
---

## d. Models Used & Evaluation Metrics Comparison

All five classification models were trained on the identical training partition and evaluated on the held-out `test_data.csv` (1,013 instances):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9191 | 0.9712 | 0.9054 | 0.8872 | 0.8962 | 0.8300 |
| **Decision Tree** | 0.9220 | 0.9264 | 0.8980 | 0.9048 | 0.9014 | 0.8369 |
| **KNN** | 0.9151 | 0.9633 | 0.8962 | 0.8872 | 0.8917 | 0.8219 |
| **Naive Bayes** | 0.8342 | 0.9444 | 0.7127 | 0.9699 | 0.8217 | 0.7014 |
| **Support Vector Machine** | 0.9220 | 0.9718 | 0.9145 | 0.8847 | 0.8994 | 0.8360 |

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Delivered strong, well-balanced baseline performance (Accuracy: `91.91%`, AUC: `0.9712`, Precision: `0.9054`). Its high ROC-AUC indicates that log-odds linear combinations of standardized word and punctuation frequencies separate most spam from legitimate mail effectively, though it misses non-linear feature interactions (such as combined capital-run spikes and currency symbols). |
| **Decision Tree** | Tied for the highest Accuracy (`92.20%`) and achieved the top F1 Score (`0.9014`) and MCC (`0.8369`). Constraining tree complexity (`max_depth=11`, `min_samples_split=8`) prevented severe overfitting while capturing non-linear threshold rules on skewed lexical frequencies. However, its AUC (`0.9264`) was the lowest among the five models because discrete leaf-node probability estimates produce coarser ranking resolution than smooth margin-based classifiers. |
| **KNN** | Achieved solid discriminative capability (Accuracy: `91.51%`, AUC: `0.9633`) using distance-weighted voting (`k=7`). Despite feature standardization, Euclidean distance in a 57-dimensional sparse lexical space suffers mildly from the curse of dimensionality, placing its F1 (`0.8917`) and MCC (`0.8219`) slightly behind SVM, Decision Tree, and Logistic Regression. |
| **Naive Bayes** | Recorded the highest Recall (`96.99%`) across all models—catching nearly every spam email—but suffered from low Precision (`71.27%`), dragging down overall Accuracy (`83.42%`) and MCC (`0.7014`). Because word frequencies and capital-letter metrics are zero-inflated and heavily correlated (violating Gaussian Naive Bayes' conditional independence and normality assumptions), the classifier over-predicts the positive spam class, causing elevated false positives. |
| **Support Vector Machine** | Tied for top Accuracy (`92.20%`) while leading all models in ROC-AUC (`0.9718`) and Precision (`91.45%`), alongside a near-top MCC (`0.8360`). The RBF kernel effectively maps non-linear combinations of character and keyword frequencies into a higher-dimensional feature space while maximizing the decision margin, minimizing false-positive classifications of legitimate emails as spam. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM)** is the overall winner for practical spam filtering on this dataset (closely contested by Decision Tree). While Decision Tree edges out SVM by `0.0009` on MCC (`0.8369` vs. `0.8360`) and `0.0020` on F1, **SVM achieves the highest AUC (`0.9718`) and highest Precision (`91.45%`)** while matching the top Accuracy (`92.20%`). In real-world email filtering, misclassifying legitimate emails as spam (false positives) is significantly costlier than letting an occasional spam email through, making SVM's superior Precision and probability calibration the strongest choice. |
