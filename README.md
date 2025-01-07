# Airline-Passenger-Prediction-Using-LSTM

## Overview
This project implements a Long Short-Term Memory (LSTM) Recurrent Neural Network in Python to predict the number of international airline passengers. The dataset used is the **International Airline Passengers dataset**, which records monthly totals of international airline passengers (in thousands) from January 1949 to December 1960.

The goal is to predict the number of passengers for a given month and year, using historical data. The project employs **TimeSeriesSplit** for model optimization and evaluates the model's performance using **Root Mean Squared Error (RMSE)**.

---

## Dataset Details
- **Source**: [International Airline Passengers Dataset](https://machinelearningmastery.com/time-series-datasets-for-machine-learning/)
- **Timeframe**: January 1949 to December 1960 (12 years, 144 observations)
- **Target**: Monthly total of international airline passengers (in units of 1,000)

---

## Project Features
- **Model**: Long Short-Term Memory (LSTM) Recurrent Neural Network
- **Optimization**: TimeSeriesSplit for hyperparameter tuning and performance evaluation
- **Error Metrics**: Root Mean Squared Error (RMSE) for train and test sets

### Results
- **Train RMSE**: 20.90
- **Test RMSE**: 46.01

---

## Repository Structure
```
.
├── data
│   └── airline_passengers.csv  # Dataset file
├── models
│   └── model.pkl           # Trained LSTM model
├── notebooks
│   └── time_series_analysis.ipynb  # Jupyter Notebook for analysis
└── README.md                   # Project description
```

---

## Implementation Steps
1. **Data Preparation**: Loaded and preprocessed the time series data, normalized values, and created sequences for training and testing.
2. **Model Design**: Designed an LSTM network to capture temporal dependencies in the data.
3. **Optimization**: Used TimeSeriesSplit for hyperparameter tuning and model selection.
4. **Training**: Trained the LSTM model on the dataset.
5. **Evaluation**: Assessed the model's performance using RMSE on both train and test sets.

---

## Usage
### Prerequisites
- Python 3.8 or higher
- Libraries: TensorFlow, NumPy, Pandas, Matplotlib, scikit-learn

### Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/SrujanBhirud/Airline-Passenger-Prediction-Using-LSTM.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Airline-Passenger-Prediction-Using-LSTM
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the notebook:
   ```bash
   jupyter notebook notebooks/time_series_analysis.ipynb
   ```

---

## Key Results and Insights
- The LSTM model effectively captures temporal patterns but demonstrates room for improvement in predicting test data.
- RMSE on the test set indicates higher error due to unseen data, suggesting the need for more robust regularization or additional features.

---

## Future Work
- Experiment with additional architectures like GRU and Transformer models.
- Incorporate external features such as economic indicators or seasonality adjustments.

---

## Acknowledgments
- Dataset provided by Jason Brownlee ([Machine Learning Mastery](https://machinelearningmastery.com/)).

