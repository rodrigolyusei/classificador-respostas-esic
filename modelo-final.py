import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix

SEED = 123

# "default" to show warnings, "ignore" to hide warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

df = pd.read_excel('./train.xlsx')
df.describe()

X = df["resp_text"].astype(str)
Y = df["clarity"]
print("X Y: ", X.shape, Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)
print("x_train x_test: ", x_train.shape, x_test.shape)
print("y_train y_test: ", y_train.shape, y_test.shape)