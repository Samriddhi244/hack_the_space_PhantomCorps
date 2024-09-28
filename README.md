# 🌞 Space Weather Prediction using Solar Data 🌍

This project focuses on predicting space weather by forecasting the **Estimated K-Index** using solar data. The **K-Index** measures geomagnetic disturbances caused by solar storms, and accurate prediction is essential for space weather forecasting, satellite communications, GPS systems, and astronaut safety.

The machine learning model in this project uses historical solar data to predict the **Estimated K-Index**, providing insights into potential geomagnetic disturbances.

---

## 📊 Dataset

The dataset used for training includes daily solar activity data with key features such as:

- **📡 Radio Flux 10.7cm**: Solar radio emissions, indicative of solar activity.
- **☀️ Sunspot Number**: The total number of sunspots on the solar surface.
- **🔬 Sunspot Area**: The total area covered by sunspots (in millionths of the solar hemisphere).
- **💥 Solar Flares (C, M, X)**: Counts of different classes of solar flares.
- **🌍 Middle Latitude A, High Latitude A, Estimated A**: Geomagnetic activity indices.
- **📉 Middle Latitude K, High Latitude K, Estimated K**: Daily K-index measures of geomagnetic activity.

### 🎯 Target Variable: **Estimated K-Index**
The **K-Index** (ranging from 0 to 9) quantifies geomagnetic activity, with higher values indicating more intense geomagnetic disturbances. 

---

## 🚀 Problem Statement

The goal is to predict the **Estimated K-Index** using solar features. This is framed as a **regression problem**, with the **K-Index** as the continuous target variable.

---

## ⚙️ Models Used

Several machine learning models were explored:

- **🌳 Random Forest Regressor**
- **🔥 Gradient Boosting Regressor**
- **🚀 XGBoost Regressor**
- **🔎 LightGBM (Light Gradient Boosting Machine)**

### 🔥 Final Chosen Model: **Gradient Boosting Regressor**

After evaluating multiple models, **Gradient Boosting Regressor** was selected due to its superior performance during cross-validation.

---

## 🏅 Model Performance

### Cross-Validation Results (MSE):

- **Random Forest**: 0.665
- **Gradient Boosting**: 0.644
- **XGBoost**: 0.691

### Final Model Performance:
- **Mean Squared Error (MSE)**: **0.644**

The **Gradient Boosting Regressor** provided the best performance, striking a balance between accuracy and computational efficiency.

![Actual vs Predicted Plot](./path-to-plot.png)

---

## 🔬 Model Development Process

### 1. **Data Preprocessing**
   - Converted features into numeric types.
   - Handled missing values through imputation.
   - Scaled the features to ensure uniformity.

### 2. **Feature Engineering**
   - Created interaction terms like combining **Sunspot Number** and **Radio Flux**.
   - Added lag features (e.g., previous day's **Radio Flux**).
   - Created rolling averages to smooth noisy data.

### 3. **Model Selection**
   - Trained baseline models (Random Forest, XGBoost).
   - Performed hyperparameter tuning using **GridSearchCV**.
   - Selected **Gradient Boosting Regressor** as the final model based on its cross-validation performance.

### 4. **Optimization**
   - Fine-tuned hyperparameters using **RandomizedSearchCV**.
   - Evaluated **LightGBM** but found **Gradient Boosting** more suitable for this dataset.

### 5. **Evaluation**
   - Plotted **Actual vs. Predicted** K-Index values to visualize the model's performance.
   - Assessed metrics like **Mean Squared Error (MSE)** and **R-squared** to quantify accuracy.

---

## 🔮 Future Work

To improve the model's accuracy, several future enhancements are planned:

1. **More Fine-Tuned Hyperparameters**: Implement **Bayesian Optimization** to further refine the model's hyperparameters.
2. **Ensemble Learning**: Use stacking with multiple models to improve accuracy.
3. **Additional Data**: Integrate more solar parameters like solar wind speed or proton flux.
4. **Deep Learning**: Explore **Recurrent Neural Networks (RNNs)** or **LSTM** networks for better handling of time-series dependencies in solar data.

---

## 💻 How to Run the Project

Follow these steps to run the project locally:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your_username/your_repository.git
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Train the Model**:
    ```bash
    python train_model.py
    ```

---

## 🎯 Conclusion

This project successfully predicts the **Estimated K-Index**, a vital measure for space weather forecasting. The model demonstrates good performance, though further fine-tuning and exploration of advanced techniques could improve its accuracy.

---

## 👥 Contributors

- **Samriddhi Bhalekar** - [GitHub Profile](https://github.com/your_username)

---

## 📝 License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE.md) file for details.
