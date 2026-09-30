# Solar Generation Condition Classification

## LearnDepth Academy LLP — Final Capstone

### Problem
Classify solar-generation operating conditions using generation and environmental measurements.

### Dataset
**Synthetic educational dataset created specifically for this capstone because an assigned dataset was not available.** It is not real plant data.

### Features
- solar_output_kW
- irradiance_W_m2
- temperature_C
- cloud_indicator
- time_of_day

### Target
generation_condition: Low / Medium / High

### Models evaluated
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree

### Results from the generated run
Best model by F1-score: **Logistic Regression**

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.996 | 0.996 | 0.996 | 0.996 |
| KNN | 0.979 | 0.979 | 0.979 | 0.979 |
| Decision Tree | 0.933 | 0.939 | 0.933 | 0.935 |

### Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Limitation
Because the dataset is synthetic, these metrics demonstrate the workflow only. Real deployment requires validated historical solar-plant data and operational testing.
