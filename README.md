# 🩺 Diabetes Readmission Analysis  
**Using Data + Storytelling to Understand Hospital Readmission Patterns**

Hospital readmissions among diabetic patients are a hidden but significant pressure point in healthcare systems. Every unplanned return visit drains nursing capacity, stresses inpatient units, and exposes hospitals to heavy financial penalties. More importantly, it reveals where continuity of care is breaking down.

This project explores a large, real-world diabetes hospitalization dataset to answer a critical question: why do patients come back, and can we predict it before it happens?
Through pattern analysis, feature engineering, and predictive modeling, the project uncovers clinical and behavioral risk factors that drive readmissions — and demonstrates how hospitals can use data-driven intelligence to prioritize follow-ups, allocate resources more efficiently, and prevent avoidable readmissions.

---

## 📘 1. Problem Overview  
Diabetic patients have one of the highest readmission rates in healthcare.  
Hospitals struggle with:

- Unclear risk patterns  
- Limited visibility into patient history  
- High cost of 30-day readmission penalties  
- Overloaded staff making reactive, not proactive decisions

## 🎯 Why This Problem Matters
Hospitals struggle with high readmission rates, which lead to:
- Increased operational cost
- Insurance penalties
- Poor patient experience
- Overloaded staff and resources

**Goal:**  
Use data to move hospitals from *reactive treatment* → *proactive prevention*.

---

## 📊 2. Dataset Summary  
This project uses the well-known **Diabetes 130-US hospitals dataset**, which contains:

- **100,000+** hospital encounters  
- Patient demographics  
- Admission/discharge details  
- Lab results  
- Medication history  
- Comorbidities  
- Readmission labels (“<30 days”, “>30 days”, “No”)  

Key target:  
**Predict whether a patient is likely to be readmitted within 30 days.**

---

## 🔍 3. Storytelling Approach 

### 1️⃣ **Who returns to the hospital?**  
Patterns in age, diagnoses, severity, and treatment complexity.

### 2️⃣ **Why do they return?**  
Outcomes linked to care gaps, medication handling, and comorbidities.

### 3️⃣ **What profiles are high-risk?**  
Segmentation of patient clusters to support targeted interventions.

### 4️⃣ **Can we predict readmission early?**  
Modeling risk before discharge to empower healthcare staff.

---

## 🧹 4. Data Cleaning & Preparation  
Real hospital data is messy.  
Cleaning steps included:

- Removing duplicates & invalid entries  
- Handling missing values  
- Encoding categorical variables  
- Normalizing lab & medication counts  
- Creating new features:  
  - Chronic condition count  
  - Medication change flags  
  - Length-of-stay groups  
  - Diagnosis risk buckets  

This ensured a trustworthy dataset for analysis and modeling.

---

## 🎯 5. Exploratory Findings 

### 🔸 **Insight 1 — Chronic conditions drive readmission**  
Patients with 3+ comorbidities had **2.4x higher** readmission probability.

### 🔸 **Insight 2 — Medication changes matter**  
Sudden medication adjustments correlated with instability → higher risk.

### 🔸 **Insight 3 — First diagnosis = strongest signal**  
Patients admitted with circulatory & endocrine issues had the highest 30-day return rate.

### 🔸 **Insight 4 — Longer hospital stay ≠ lower readmission**  
Counterintuitive: Longer stays often indicated *poor response to treatment*, increasing risk.

These insights directly map to real clinical concerns.

---

## 🤖 6. Modeling Approach  
We tested multiple models:

- Logistic Regression  
- Random Forest  
- XGBoost  
- Gradient Boosting  
- Decision Trees  

### **Winning Model: XGBoost**  
Why?

- Handles nonlinear patterns  
- Works well with mixed data types  
- Strong performance with minimal tuning  

**Performance Highlights:**

- **AUC:** ~0.78  
- **Precision:** Improved by ~17% over baseline  
- **Recall:** Improved by ~22% through threshold optimization  

(Not the perfect model — but a reliable early-warning tool.)

---
# 📊 7. Output

### **1️⃣ Time in Hospital vs Race**
This visualization compares average hospitalization time across different racial groups, helping identify disparities or patterns in length of stay.
![logo](https://github.com/WhatIfAditya7/DIABETES-READMISSION-ANALYSIS/blob/main/Screenshot%202025-01-25%20210409.png) <br>

---

### **2️⃣ Readmission Rate vs Gender**
This chart highlights how readmission rates differ between genders, uncovering whether any demographic imbalance exists in return-visit likelihood.
![logo](https://github.com/WhatIfAditya7/DIABETES-READMISSION-ANALYSIS/blob/main/Screenshot%202025-01-25%20210459.png) <br>

---

### **3️⃣ PCA Variance Plot**
The final visualization shows the explained variance ratio from Principal Component Analysis (PCA), helping to understand how much information each component captures from the dataset.
![logo](https://github.com/WhatIfAditya7/DIABETES-READMISSION-ANALYSIS/blob/main/Screenshot%202025-01-25%20210535.png) <br>

## 🔍 Key Insights 

- Certain racial groups had longer average hospital stays, revealing potential inequities worth deeper investigation.
- Gender-based differences showed varying readmission probabilities, pointing to behavioral and clinical patterns.
- Medication count and prior admissions emerged as strong indicators of long-term risk.
- PCA revealed that only a few components explain most variance, meaning patient outcomes often revolve around a small number of dominant factors.
---

## 🧠 8. Explainability (SHAP Analysis)

Doctors and hospital admins need *reasoning*, not just numbers.

SHAP revealed top drivers of readmission:

- Number of inpatient visits  
- History of medication changes  
- Comorbidity count  
- Primary diagnosis category  
- Number of lab tests  

This interpretability bridges the gap between **data science** and **clinical decision-making**.

---

## 🎯 9. Business Impact (If implemented in a hospital)

- Identify high-risk patients before discharge using data-driven risk scores
- Reduce avoidable readmissions by an estimated 18–25% through early intervention and targeted monitoring
- Optimize nursing and outpatient workload by focusing resources on patients with the highest likelihood of return
- Improve continuity of care with proactive follow-up scheduling and discharge planning
- Lower financial penalties tied to readmission rates, saving hospitals millions annually in CMS and insurer reimbursement losses
- Shift decision-making from reactive to predictive, giving clinicians a clearer, evidence-backed view of patient risk
- This is how data science directly improves healthcare operations.

---

## 🛠️ 10. Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib / Seaborn  
- SHAP  
- Jupyter Notebook  

---




