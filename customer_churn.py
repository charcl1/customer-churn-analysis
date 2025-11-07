import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve, roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn import set_config
set_config(display="diagram")
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("./Telco-Customer-Churn.csv")
df.head()

df.info()

df.describe()

null_counts = df.isnull().sum()
print(null_counts)

# preprocessing data
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan).astype(float)
df = df.dropna()
df['TotalCharges'].astype('double')
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Select features by type (categorical or numerical)
SEED = 365
TARGET = 'Churn'
FEATURES = df.columns.drop(TARGET)

NUMERICAL = df[FEATURES].select_dtypes('number').columns
print(f"Numerical features: {', '.join(NUMERICAL)}")

CATEGORICAL = pd.Index(np.setdiff1d(FEATURES, NUMERICAL))
print(f"Categorical features: {', '.join(CATEGORICAL)}")

df.head()

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

X = df.drop(columns=TARGET)
Y = df[TARGET]

scaler = MinMaxScaler()
num_scaled = scaler.fit_transform(X[NUMERICAL])

encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
cat_encoded = encoder.fit_transform(X[CATEGORICAL])

preprocessed = np.concatenate((num_scaled,cat_encoded), axis=1)

columns = np.append(NUMERICAL, encoder.get_feature_names_out(CATEGORICAL))
preprocessed = pd.DataFrame(preprocessed, columns=columns, index=X.index)

preprocessed.head()

preprocessed.info()

# train test split
train_preprocessed, test_preprocessed, y_train, y_test = train_test_split(preprocessed, Y, test_size=0.2, random_state=SEED)

# Set up a logistic regression model type to train
model = LogisticRegression()
model.fit(train_preprocessed, y_train)

probs = model.predict_proba(test_preprocessed)[:,-1] # get the probability of y=1
predictions = model.predict(test_preprocessed) # binary predictions

y_proba = model.predict_proba(test_preprocessed)[:,1]
auc_score = roc_auc_score(y_test, y_proba)
print(f"AUC Score: {auc_score}")

fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# create the confusion matrix for the model
confusion_matrix(y_test, predictions)

# accuracy scores
predictions_test = predictions
accuracy_test = accuracy_score(y_test, predictions_test)
predictions_train = model.predict(train_preprocessed)
accuracy_train = accuracy_score(y_train, predictions_train)
print("Train accuracy is = ", accuracy_train)
print("Test accuracy is = ", accuracy_test)

sns.boxplot (y=probs, x=y_test);
plt.xlabel("Actually Churn")
plt.ylabel("Probability of Leaving")
plt.title("Model Predicted Probability to Churn")
plt.show()