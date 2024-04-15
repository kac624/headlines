# core
import json
import torch
import time
import joblib
import os

# modeling
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# evaluation
from sklearn.metrics import accuracy_score, f1_score



"""SETUP"""

start = time.time()
os.makedirs('models', exist_ok=True)



"""CONFIG"""

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

with open('scripts/_config.json', 'r') as file:
    config = json.load(file)


"""LOAD DATA"""

tfidf_train = torch.load('data/tfidf_train.pt')
tfidf_valid = torch.load('data/tfidf_valid.pt')
y_train = torch.load('data/y_train.pt')
y_valid = torch.load('data/y_valid.pt')


"""LOGISTIC REGRESSION"""

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(tfidf_train, y_train)

y_pred_train = lr_model.predict(tfidf_train)
y_pred_valid = lr_model.predict(tfidf_valid)

print(
    f'\n--Logistic Regression Results--'

    f'\nTrain Accuracy: {100 * accuracy_score(y_train, y_pred_train):.2f}%'
    f'\nTrain F-1 Score: {100 * f1_score(y_train, y_pred_train, average="weighted"):.2f}%'

    f'\nValidation Accuracy: {100 * accuracy_score(y_valid, y_pred_valid):.2f}%'
    f'\nValidation F-1 Score: {100 * f1_score(y_valid, y_pred_valid, average="weighted"):.2f}%'

    f'\nTime Elapsed: {(time.time() - start)/60:.2f} minutes'
)


"""XGBOOST"""

xgb_model = XGBClassifier(tree_method='hist', device='cuda')
xgb_model.fit(tfidf_train, y_train)

y_pred_train = xgb_model.predict(tfidf_train)
y_pred_valid = xgb_model.predict(tfidf_valid)

print(
    f'\n--XGBoost Results--'
    f'\nTrain Accuracy: {100 * accuracy_score(y_train, y_pred_train):.2f}%'
    f'\nTrain F-1 Score: {100 * f1_score(y_train, y_pred_train, average="weighted"):.2f}%'

    f'\nValidation Accuracy: {100 * accuracy_score(y_valid, y_pred_valid):.2f}%'
    f'\nValidation F-1 Score: {100 * f1_score(y_valid, y_pred_valid, average="weighted"):.2f}%'

    f'\nTime Elapsed: {(time.time() - start)/60:.2f} minutes'
)


"""SAVE MODEL"""

joblib.dump(lr_model, 'models/lr_model.pkl')
xgb_model.save_model('models/xgb_model.json')


"""END"""

print(f'Total Time: {(time.time() - start)/60:.2f} minutes')