# Smart Room Intelligence: AI-Powered Hotel Booking Cancellation Predictor

## Project Overview

This project focuses on developing an intelligent machine learning system that predicts hotel booking cancellations using structured reservation data. The solution is designed as a core component of Nexuso' smart room platform—delivering **real-time personalization** and **operational automation** by forecasting guest behavior.

By anticipating cancellations, the system enables smarter energy usage, optimized room assignments, and staff scheduling, contributing to the creation of **smarter hotel spaces** that respond dynamically to guest preferences.

---

##  Goal

To build a predictive engine that forecasts the likelihood of booking cancellations using guest, reservation, and behavioral data—enabling proactive decision-making for staff, smarter room control, and a personalized guest experience through the Nexus platform.

---

##  Intended Audience

- Hotel operations teams seeking automation  
- AI engineers developing real-time predictive services  
- Product teams building intelligent hospitality apps  
- Investors and decision-makers evaluating MVP intelligence layer  

---

##  Strategy & Pipeline Steps

###  Preprocessing
- Loaded hotel booking data (`hotel_bookings.csv`)
- Combined arrival dates and encoded categorical features
- Cleaned missing values, created engineered fields (`total_guests`, `stay_length`)
- Encoded `is_canceled` as target

### EDA & Feature Importance
- Top predictors: **lead time** and **deposit type**
- Cancellation spikes in summer and long-lead-time bookings

### Modeling with Random Forest
- Used `RandomForestClassifier` for performance and interpretability
- Used SMOTE for class balance
- Tuned hyperparameters via `GridSearchCV`
- Accuracy: **~86%**, ROC-AUC: **0.88**

###  Streamlit Web App
- Live input form for lead time, deposit type, market segment, guests
- Output: ❌ *Likely to Cancel* / ✅ *Safe Booking*
- SHAP explainability available for transparency

---

##  Challenges
- Class imbalance handled via SMOTE  
- Dropped high-cardinality features (e.g., `agent`, `company`)  
- Imputed missing values (e.g., `children`, `country`)  

---

##  Problem Statement

Can Nexuso use AI to accurately predict which hotel bookings are likely to be canceled, allowing staff to automate fallback operations, and dynamically personalize guest journeys?

---

## 🗃 Dataset

- Source: [Hotel Booking Demand Dataset – Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)  
- 119,000+ records with guest demographics and reservation metadata  
- Target: `is_canceled` (binary classification)

---

##  Implementation Overview

- Python + Scikit-learn for modeling  
- SHAP for feature importance  
- Streamlit for user interaction  
- GitHub + Streamlit Cloud for deployment  

---

##  Streamlit App Features

- Interactive form to simulate new bookings  
- Live classification: *Canceled / Not Canceled*  
- Feature breakdown shown via visual bar plot  
- Mobile-ready app for staff dashboard  

---

##  Visualizations & Results

- Confusion Matrix (Precision: 0.82, Recall: 0.75)  
- ROC-AUC Score: **0.88**  
- Most influential features:
  - Lead Time  
  - Deposit Type  
  - Total Guests  
  - Market Segment

---

##  Conceptual Enhancement

**LangChain + RAG Q&A for Policy/Support**

Allow users to ask:
> “What’s the average cancellation risk for bookings longer than 10 days in city hotels?”

Answers powered by RAG pipeline querying a vector store connected to booking records and staff policy documents.

---

##  References

- https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand  
- https://docs.streamlit.io/  
- https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html  
- https://github.com/shap/shap  
