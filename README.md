# Air Quality Index Prediction: README

## Overview

This project aims to predict the Air Quality Index (AQI) for Delhi, India, using machine learning techniques. The project uses weather-related data as features and AQI data as labels to train the model. The weather data is obtained through web scraping, and the AQI data is sourced from Kaggle.

## Features

- Data Collection: Web scraping for weather data
- Data Preprocessing: Handling missing values, data transformation
- Data Visualization: Correlation matrix, time-series plots
- Model Building: Random Forest Regressor, Linear Regression
- Model Evaluation: MAE, MSE, RMSE, Cross-Validation
- Hyperparameter Tuning: RandomizedSearchCV

## Requirements

- Python 3.x
- Google Colab (for running the notebook)
- Libraries: pandas, matplotlib, seaborn, scikit-learn, BeautifulSoup, requests

## Installation

1. Clone the repository
    ```bash
    git clone https://github.com/priyamvada21/AQI_Prediction.git
    ```
2. Install the required packages
    ```bash
    pip install pandas matplotlib seaborn scikit-learn requests beautifulsoup4
    ```

## How to Run

1. Open the `AQI_Prediction.ipynb` notebook in Google Colab.
2. Run all the cells to collect data, preprocess it, and train the model.

## Data Collection

- Weather data is scraped from [Tutiempo](http://en.tutiempo.net/) for the years 2015-2019.
- AQI data is downloaded from [Kaggle](https://www.kaggle.com/rohanrao/air-quality-data-in-india).

## Data Preprocessing

- Missing values are handled.
- Data is transformed to numerical format.

## Model Building

- Random Forest Regressor is used for prediction.
- Linear Regression is also implemented for comparison.

## Model Evaluation

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Cross-Validation Score

## Hyperparameter Tuning

- RandomizedSearchCV is used for hyperparameter tuning of the Random Forest model.

## Files

- `AQI_Prediction.ipynb`: Main notebook containing all the code
- `city_day.csv`: AQI data for various cities
- `Complete_dataframe.csv`: Preprocessed data used for training the model
- `AQI_random_forst_model.pkl`: Saved Random Forest model
- `AQI_random_forst_RandomSearchCV.pkl`: Saved Random Forest model with hyperparameter tuning

## Future Work

- Implement more machine learning algorithms for comparison.
- Extend the project to predict AQI for other cities.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for the AQI dataset
- [Tutiempo](http://en.tutiempo.net/) for the weather data

---

**Happy Coding!**
