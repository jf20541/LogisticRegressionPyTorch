# LogisticRegression

## Objective
Logistic Regression using Pytorch and from scratch to determine the impact of multiple independent variables presented simultaneously to predict binary target values [1: RainTomorrow, 0:No RainTomorrow]

## Repository File Structure
    ├── src          
    │   ├── pytorchmodel.py      # Logistic Regression using PyTorch and evaluate metric
    │   ├── models.py            # Logistic Regression from scratch
    │   ├── train.py             # Initiated the model, evaluate metric and initializing Argument Parser Class
    │   ├── create_folds.py      # Implemented a cross-validation set
    │   ├── data.py              # Cleaned the data and feature engineer
    │   └── config.py            # Define path as global variable
    ├── inputs
    │   ├── train.csv            # Training dataset
    │   └── train_folds.csv      # K-Fold dataset LR_fold0.bin
    ├── models                   # Saving/Loading models parameters
    │   ├── LR_fold0.bin
    │   ├── LR_fold1.bin
    │   ├── LR_fold2.bin 
    │   ├── LR_fold3.bin 
    │   └── LR_fold4.bin
    ├── requierments.txt         # Packages used for project
    └── README.md
    


## Model
Supervised-Learning method for binary classification. It uses a sigmoid function **(σ)** to model a curve where the predictor domain features can be conditional probability between [0,1]. Logistic refers to the log-odds probability model, which is the ratio of the probability that an event occurs to the probability that it doesn't occur, given in the equation below.


![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B200%7D%20%5Cfn_phv%20%5Cbeta%20X%20%3D%20log%5Cleft%20%28%5Cfrac%7Bp%28x%29%7D%7B1-p%28x%29%7D%20%5Cright%20%29)\
![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B200%7D%20%5Cfn_phv%20%5Csigma%20%3D%20p%28X%29%20%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20e%5E%7B%5E%7B-%5Cbeta%20X%7D%7D%7D)\


## Metric & Mathematics
It uses the **Maximum Likelihood Estimation (MLE)** to find the optimal parameters. For labels [0, 1] it estimates parameters such that the product of all conditional probabilities of class [0,1] samples are as close to maximum value [0,1].

![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B200%7D%20%5Cfn_phv%20L%28%5Cbeta%20%29%20%3D%20%5Cprod_%7Bs%5C%2C%20in%20%5C%2C%20y_%7Bi%7D%3D1%7D%5E%7B%7D%20p%28x_%7Bi%7D%29%20*%20%5Cprod_%7Bs%5C%2C%20in%20%5C%2C%20y_%7Bi%7D%3D0%7D%5E%7B%7D%20%281-p%28x_%7Bi%7D%29%29)

Combine the products, take the log-likelihood, and convert into summation
![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B200%7D%20%5Cfn_phv%20l%28%5Cbeta%20%29%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20y_%7Bi%7D%5C%2C%20log%28p%28x_%7Bi%7D%29%29%20&plus;%20%281-y_%7Bi%7D%29log%281-p%28x_%7Bi%7D%29%29)


Substitute p(x_1) with it's exponent form, group the coefficients of y_i and simplify to optimize beta coefficient that maximizes this function. 

![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B200%7D%20%5Cfn_phv%20l%28%5Cbeta%20%29%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20y_%7Bi%7D%5Cbeta%20x_%7Bi%7D%20-%20log%281%20&plus;%20e%5E%7B%5Cbeta%20x_%7Bi%7D%7D%29)

## Output
```
Epoch 500 and  Loss: 15.6091
Logistic Regression using Pytorch Accuracy: 78.15%
```
```
Log Loss for Logistic Regression: 0.779 for Fold=0
Log Loss for Logistic Regression: 0.779 for Fold=1
Log Loss for Logistic Regression: 0.777 for Fold=2
Log Loss for Logistic Regression: 0.779 for Fold=3
Log Loss for Logistic Regression: 0.781 for Fold=4
```
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
