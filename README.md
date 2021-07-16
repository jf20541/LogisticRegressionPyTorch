# LogisticRegression

## Objective
Logistic Regression using Pytorch and from scratch to determine the impact of multiple independent variables presented simultaneously to predict binary target values [1: RainTomorrow, 0:No RainTomorrow]

## Model
![](https://latex.codecogs.com/gif.latex?sigmoid%20%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20e%5E%7B-y_%7Bi%7D%7D%7D)\
![](https://latex.codecogs.com/gif.latex?y_%7Bi%7D%20%3D%20%5Cbeta%20_%7B0%7D%20&plus;%20%5Cbeta%20_%7B1%7DX_%7B1%2Ci%7D&plus;%20...%20&plus;%20%5Cbeta_%7Bk%7DX_%7Bk%2Ci%7D%2C%20i%3D1%2C....%2C%20n)\
![](https://latex.codecogs.com/gif.latex?Logistic%20Regression%20%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20e%5E%7B-%28%5Cbeta%20_%7B0%7D%20&plus;%20%5Cbeta%20_%7B1%7DX_%7B1%2Ci%7D&plus;%20...%20&plus;%20%5Cbeta_%7Bk%7DX_%7Bk%2Ci%7D%29%7D%7D)


## Metric & Mathematics

![](https://latex.codecogs.com/gif.latex?Decision%20Boundary%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%201%20%5C%3Bif%5C%3B%20P%28y%3D1%7Cx%29%3E0.5%5C%5C0%5C%3B%20%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%5C%3B%5C%3Botherwise%20%5Cend%7Bmatrix%7D%5Cright.)
- `Cross Entropy (Log Loss)`:\
![](https://latex.codecogs.com/gif.latex?LogLoss%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5By%5E%7Bi%7Dlog%28h_%7B0%7D%28x%5E%7Bi%7D%29%29%20&plus;%20%281-y%5E%7Bi%7D%29log%281-h_%7B0%7D%28x%5E%7Bi%7D%29%29%5D)


## Output
```bash
Epoch 500 and  Loss: 15.6091
Logistic Regression using Pytorch Accuracy: 78.15%
```
```
```bash
Log Loss for Logistic Regression: 0.779 for Fold=0
Log Loss for Logistic Regression: 0.779 for Fold=1
Log Loss for Logistic Regression: 0.777 for Fold=2
Log Loss for Logistic Regression: 0.779 for Fold=3
Log Loss for Logistic Regression: 0.781 for Fold=4
```

### Code
- `pytorchmodels.py`: Logistic Regression using PyTorch and evaluate metric
- `models.py`: Logistic Regression from scratch
- `train.py`: Initiated the model, evaluate metric and initializing Argument Parser Class
- `create_folds.py`: Implemented a cross-validation set
- `config.py`: Define path as global variables
- `data.py`: Cleaned the data and feature engineer

## Data Features and Target
[Kaggle's Weather Data](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)
```bash
Target
RainTomorrow    float64

Features
MinTemp         float64
MaxTemp         float64
Rainfall        float64
Humidity9am     float64
Humidity3pm     float64
Pressure9am     float64
Pressure3pm     float64
Temp9am         float64
Temp3pm         float64
RainToday       float64
Year              int64
Month             int64
```

## Sources
https://www.kaggle.com/jsphyg/weather-dataset-rattle-package \
http://www.bom.gov.au/climate/data. \
https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
