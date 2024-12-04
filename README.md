# Deliverables/Documentation

### Website
[https://yanni-ma.github.io/EPI-Project/](https://yanni-ma.github.io/EPI-Project/)

### Project Deliverables
- **Project Code - Full Repo:**  
  [GitHub Repository](https://github.com/yanni-ma/EPI-Project)

- **Project Code - Tarball:**  
  [Tarball](https://github.com/yanni-ma/EPI-Project/blob/main/tar_submission.tar.gz)

- **Project Poster:**  
  [Google Slides Poster](https://docs.google.com/presentation/d/1V-Z6607hsw5ShxEzHhPr3N0p2Z__Mo120p9noSjtQEQ/edit?usp=sharing)

- **Project Final Report:**  
  [Final Report](https://github.com/yanni-ma/EPI-Project/blob/main/tar_submission/DOC/CSE_8803_Project_Final_Report.pdf)

# Political SSEs Impact on COVID-19 Case Count Analysis and Forecasting 

This repository contains Python scripts for analyzing and forecasting COVID-19 case counts using advanced statistical models like SARIMAX and SEIR. The project incorporates event-based data (valence, size_mean) to study the potential impacts of political events on case trends.

## important files

- **`SEIR_with_counties.py`**: Simulates the spread of infections using the SEIR model, incorporating county-level data
- **`daily_arima.py`**: Fits ARIMA models with smoothing to forecast case counts
- **`daily_arima_without_smoothing.py`**: Fits ARIMA models without smoothing for comparison
- **`data_preprocessing.py`**: Contains helper functions for loading and preprocessing datasets
- **`main.py`**: File for analyzing and visualizing case trends with basic forecasting
- **`minimize_RMSE_for_smoothed_ARIMA.py`**: Grid search to minimize RMSE for ARIMA models with smoothing
- **`minimize_RMSE_smoothed_no_exog.py`**: Grid search to minimize RMSE for ARIMA models without exogenous variables
- **`modified_main_to_detect_spikes.py`**: Script for detecting spikes in case counts from preprocessed data
- **`unweighted_arima_valence_case_counts_projection.py`**: ARIMA forecasting considering valence without weighting by event size
- **`weighted_arima_valence_case_counts_projection.py`**: ARIMA forecasting considering valence, weighted by event size
- **`measure_arima_performance.py`**: Compares performance metrics (e.g., RMSE) of different ARIMA models

## Datasets

The datasets are provided in the dataset folder. They are sourced from NYT (https://github.com/nytimes/covid-19-data) and CCC (https://github.com/nonviolent-action-lab/crowd-counting-consortium)

## Usage

To run any Python script, use the following command in your terminal while cd'd into the SRC folder:

```bash
python <script_name>.py
