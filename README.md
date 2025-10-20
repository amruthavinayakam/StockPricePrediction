# 📈 Predicting Stock Price Movements Using Machine Learning: A Comparative Analysis

## 🧾 Abstract

In the fast-paced landscape of entrepreneurship and finance, successful decision-making depends on understanding complex market dynamics. Traditional analysis often fails to capture subtle behavioral and data-driven trends. This research leverages **supervised and unsupervised machine learning algorithms**—including clustering and classification models—to identify hidden patterns in stock market data and enhance predictive accuracy.

The study specifically focuses on **Intel Corporation (INTC)** and compares the performance of several machine learning models under two trading strategies. The goal is to empower investors and analysts with **data-driven insights** that enable smarter trading and risk management.

---

## 🎯 Objectives

* Predict stock price movements using supervised ML algorithms.
* Compare multiple classification models across two trading strategies.
* Evaluate model performance using accuracy, RMSE, MAE, MSE, and Sharpe Ratio.
* Identify the most efficient predictive model and trading approach.

---

## 🧠 Introduction

Predicting stock market movements has long been a challenge due to the volatile and nonlinear nature of financial data. Traditional approaches—fundamental and technical analysis—often miss deeper data relationships. Machine learning enables models to **learn from historical trends**, identify correlations, and **forecast future price behavior** more accurately.

---

## 📚 Background

Recent advancements in machine learning have introduced algorithms capable of processing complex financial datasets. Methods such as **Support Vector Machines (SVMs)**, **Random Forests**, and **Neural Networks** have been applied successfully to model dependencies and relationships between price indicators, volatility, and volume trends.

By combining classical statistical models with deep learning, researchers can enhance both interpretability and predictive power.

---

## ⚙️ Methodology

The study employs two strategies for Intel Corp (INTC):

### **Strategy 1 — Next-Day Price Prediction**

Predict whether the next trading day's closing price will be higher or lower than the current day's close.

### **Strategy 2 — Moving Averages**

Compare the **50-day** and **200-day** moving averages to identify **bullish** and **bearish** signals.

---

### **Steps Followed**

1. **Data Acquisition:** Historical INTC data via the Yahoo Finance API.
2. **Preprocessing:** Cleaning missing values, formatting, adjusting for splits/dividends.
3. **Feature Engineering:** Deriving technical indicators and ratios.
4. **Label Generation:** Assigning "up" or "down" classes based on strategy.
5. **Model Selection:** Implementing ML classifiers and deep learning models.
6. **Evaluation:** Using test datasets and computing metrics (accuracy, RMSE, Sharpe Ratio).

---

## 🤖 Models Used

* **K-Nearest Neighbors (KNN):** Classifies based on nearest data points.
* **Random Forest (RF):** Ensemble of decision trees; handles noise and complexity.
* **Gradient Boosting (GB):** Sequential tree-based learner improving residuals.
* **Support Vector Machines (SVMs):** Finds optimal hyperplane for classification.
* **XGBoost:** Optimized gradient boosting with regularization.
* **LSTM (Long Short-Term Memory):** Captures sequential dependencies and time lags.

Each algorithm was fine-tuned using **grid search** and **cross-validation** for hyperparameter optimization.

---

## 🧩 Implementation

**Data Retrieval:**
Historical data for INTC (Open, Close, High, Low, Volume) retrieved via Yahoo Finance API.

**Preprocessing:**
Data cleaning, normalization, and feature scaling performed to ensure consistency.

**Exploratory Data Analysis (EDA):**

* Trends and volatility visualization
* Moving averages and correlations
* Volume and return distribution
* Identification of anomalies or regime shifts

---

## 📊 Model Results

**Strategy 1 (Next-Day Price Prediction)**

* Random Forest: 0.51
* XGBoost: 0.66
* KNN: 0.55

**Strategy 2 (Moving Averages)**

* KNN: 0.98
* Random Forest: 0.99
* Gradient Boosting: 0.99
* SVM: 0.97
* XGBoost: 0.92

**Observation:**
Strategy 2, which relies on moving averages, consistently outperformed next-day prediction methods due to reduced volatility noise and better trend representation.

---

## 📈 Performance Metrics

**K-Nearest Neighbors (KNN)**
RMSE: 0.2459
MAE: 0.0302
MSE: 0.0605
Sharpe Ratio: 0.0195

**Random Forest**
RMSE: 0.2008
MAE: 0.0202
MSE: 0.0403
Sharpe Ratio: 0.0185

**Gradient Boosting**
RMSE: 0.2008
MAE: 0.0202
MSE: 0.0403
Sharpe Ratio: 0.0662

**SVM**
RMSE: 0.3619
MAE: 0.0655
MSE: 0.1310
Sharpe Ratio: 0.2300

**XGBoost**
RMSE: 0.1420
MAE: 0.0202
MSE: 0.0202
Sharpe Ratio: 0.0455

**LSTM**
RMSE: 0.5605
MAE: 0.0146
MSE: 0.1568
Sharpe Ratio: -0.0336

---

## 🧩 Key Insights

* **SVM achieved the best risk-adjusted performance** (highest Sharpe Ratio).
* **XGBoost achieved the best prediction accuracy** (lowest RMSE).
* **LSTM underperformed**, indicating it may require deeper tuning or more data.
* **Strategy 2 (Moving Averages)** proved more effective and realistic for trading decisions, smoothing out daily noise.

---

## 📉 Visual Analyses

Visualizations included in this study:

* High vs. Low Stock Prices
* Open vs. Close Trends
* Volume Over Time
* Correlation Matrix
* 30-Day Moving Average vs. Closing Price
* Daily Return Percentage
* 50-Day vs. 200-Day Moving Average
* True vs. Predicted Values

---

## 🔮 Future Work

Future improvements include:

* Fine-tuning LSTM architectures and hyperparameters.
* Introducing **attention mechanisms** for temporal focus.
* Implementing a **critique agent** using reinforcement learning for feedback-based optimization.
* Integrating external features (macroeconomic data, sentiment analysis).

---

## 🧾 References

1. Masoud, N.M.H. (2017). *The impact of stock market performance upon economic growth.*
2. Murkute, A. & Sarode, T. (2015). *Forecasting market price of stock using artificial neural network.*
3. Li, L. et al. (2017). *Research on machine learning algorithms and feature extraction for time series.*
4. Huang, W. et al. (2005). *Forecasting stock market movement direction with support vector machine.*
5. Selvin, S. et al. (2017). *Stock price prediction using LSTM, RNN, and CNN sliding window models.*
6. Kumar, M. & Thenmozhi, M. (2006). *Forecasting stock index movement: A comparison of SVM and Random Forest.*
7. Yahoo Finance API – Data Source for Intel Corp (INTC).

---

## 🧰 Tech Stack

* **Languages:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow/Keras, XGBoost, Matplotlib, Seaborn
* **Data Source:** Yahoo Finance API

---

## 📜 License

This project is released under the **MIT License**.
Feel free to fork, modify, and contribute.

