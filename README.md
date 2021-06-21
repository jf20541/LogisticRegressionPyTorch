# LogisticRegression

## Objective


## Metric
- `Area Under the Receiver Operating Characteristic Curve (ROC AUC)`:
- `Accuracy`:

## Output
```bash
sh run.sh 
```
```bash
Accuracy for Logistic Regression: 0.779 and Fold=0
Accuracy for Logistic Regression: 0.779 and Fold=1
Accuracy for Logistic Regression: 0.777 and Fold=2
Accuracy for Logistic Regression: 0.779 and Fold=3
Accuracy for Logistic Regression: 0.781 and Fold=4
```

### Code
Created 5 modules
- `models.py`: create a Logistic Regression from scratch
- `train.py`: Initiated the model, evaluated the models metric and initializing Argument Parser Class
- `create_folds.py`: Implemented a cross-validation set
- `config.py`: Define path as global variables
- `data.py`: Cleaned the data and added more varaibles

### Run
In a terminal or command window, navigate to the top-level project directory `MonteCarloPorfolioOptimization/` (that contains this README) and run the following command:
```bash
pip install --upgrade pip && pip install -r requirements.txt && sh run.sh
``` 

## Data Features and Target
[Kaggle's Weather Data](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)\
Target
- `RainTomorrow`:

Features
- `MinTemp`: 
- `MaxTemp`: 
- `Rainfall`: 
- `Humidity9am`: 
- `Humidity3pm`: 
- `Pressure9am`: 
- `Pressure3pm`: 
- `Temp9am`: 
- `Temp3pm`: 
- `RainToday`: 
- `Year`: 
- `Month`: 

## Sources
https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
http://www.bom.gov.au/climate/data.
