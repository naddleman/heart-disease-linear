"""A linear predictor for heart disease from the UCI Clevland dataset

'thal' refers to thalassemia
"""

import numpy as np
import pandas as pd

FILENAME = "processed.cleveland.data"
COLUMNS = ['age',       #years
           'sex',       #1 = male, 0 = female
           'cp',        #chest pain type: 1,2,3,4
           'trestbps',  #resting blood pressure (mmHg)
           'chol',      #cholesterol mg/dl
           'fbs',       #fasting blood sugar > 120 mg/dl (1/0:T/F)
           'restecg',   #resting ecg 0:normal 1: abnormal 2: ventricual abn
           'thalach',   #max heart rate achieved
           'exang',     #exercise induced angina (T/F)
           'oldpeak',   #ST depression induced by exercise
           'slope',     # slope of peak exercise ST segment 1,2,3 categorical
           'ca',        # count colored vessesl by fluoroscopy
           'thal',      #3 = normal; 6 = fixed defect; 7=reversable defect
           'num']       # target variable: 0: <50% narrowing >0: >50% narrowing

df = pd.read_csv(FILENAME, names=COLUMNS)

# The UCI dataset uses '?' to represent missing values.
df[df.eq('?').any(1)].index 

# drop these rows
df = df[~df[df == '?'].any(1)]

# We normalize the real/integer valued columns (i.e., not categorical/ordinal)
rv_cols = ['age','trestbps','chol','thalach','oldpeak']
df_rv = df[rv_cols]
df_rv = (df_rv - df_rv.mean()) / np.sqrt(df_rv.var()) # z-scores

## NB: Check these for normality! They are not all normal

#df[rv_cols] = df_rv

#Make dummy variables
df_cp = pd.get_dummies(df.cp,prefix='cp')
df_re = pd.get_dummies(df.restecg,prefix='restecg')
df_slope = pd.get_dummies(df.slope,prefix='slope')
df_thal = pd.get_dummies(df.thal,prefix='thal')
df2 = df.join([df_cp,df_re,df_slope,df_thal])
cols = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach',
       'exang', 'oldpeak', 'ca', 'num', 'cp_1.0', 'cp_2.0',
       'cp_3.0', 'cp_4.0', 'restecg_0.0', 'restecg_1.0', 'restecg_2.0',
       'slope_1.0', 'slope_2.0', 'slope_3.0', 'thal_3.0', 'thal_6.0',
       'thal_7.0']

# We do not need the redundant dummy variable columns
# In fact they will hurt us as they will not be linearly independent
# And so our feature matrix will not have full rank (and the regression
# may end up with huge coefficients)

cols = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach',
       'exang', 'oldpeak', 'ca', 'num', 'cp_1.0', 'cp_2.0',
       'cp_3.0', 'restecg_0.0', 'restecg_1.0', 
       'slope_1.0', 'slope_2.0', 'thal_3.0', 'thal_6.0']

df2 = df2[cols]
df2[rv_cols] = df_rv

# Make a matrix where the rows are inputs (each with first element 1)
# and the columns are the various attributes, and a vector of outputs

X = np.array(df2.drop('num', axis=1), dtype=float)
X = np.hstack((np.ones((len(X),1)),X))
features = ['intercept'] + cols[:9] + cols[10:] #
y = np.array(df2.num, dtype=float)
y2 = np.array(y>=1,dtype=float)

# The residual sum of squares is given by the formula
# (y - XB).T (y - XB) where
# y is the vector of targets, X is the N x (p + 1) matrix with rows input
# vectors, and parameters of the linear equation, B.
# We minimize this in the usual fashion, differntiating wrt B, and setting
# the derivative to zero:
# -2X.T(y-XB) = 0 -> X.T (y - XB) = 0 -> X.T y - X.T XB = 0
# -> X.T XB = X.T y -> B_hat = (X.T X)^(-1) X.T y
# So predicted values y_hat = XB_hat = X(X.T X)^(-1) X.T y


#h = np.linalg.inv(X.T @ X) @ X.T
#Beta = h @ y # Regression coefficients
#Beta2 = h @ y2
#H = X @ h # The so-called "hat" matrix, because it puts the hat on y

# Split training and testing sets
np.random.seed(101)
shuffled = list(range(X.shape[0]))
np.random.shuffle(shuffled)
training_size = X.shape[0] * 4 // 5
train_idx, test_idx = shuffled[:training_size], shuffled[training_size:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y2[train_idx], y2[test_idx]

Beta_trained = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

coeffs_dict = {k:v for k,v in zip(features,Beta_trained)}

def misses(X,y,Beta):
    return np.abs(y - ((X @ Beta) > 0.5)).sum() 

train_misses = misses(X_train, y_train, Beta_trained)
test_misses  = misses(X_test, y_test, Beta_trained)
train_hits = X_train.shape[0] - train_misses
test_hits  = X_test.shape[0] - test_misses

print("Training set classification accuracy:\n" \
      f"{int(train_hits)}/{X_train.shape[0]} identified: "\
      f"{train_hits / X_train.shape[0] * 100:.2f}%\n"\
      "Test set classification accuracy:\n" \
      f"{int(test_hits)}/{X_test.shape[0]} identified: "\
      f"{test_hits / X_test.shape[0] * 100:.2f}%")

# We see the simplest linear classifier does quite well for this data
# We may compute Z scores for the coefficients to determine whether they have
# a significant effect.

# The Z score for the j'th coefficient is computed
# z_j = Beta_j / (stdv * sqrt(v_j)
# Where v_j = (X_T @ X)^(-1) _j

var = (np.sum((y_train - (X_train @ Beta_trained))**2) /
        (X_train.shape[0] - X_train.shape[1] - 1))
sigma = np.sqrt(var)
var_covar = np.diag(np.linalg.inv(X_train.T @ X_train))

z_scores = Beta_trained / (sigma * var_covar)
z_dict = {k:v for k,v in zip(features, z_scores)}
linear_fit = pd.DataFrame.from_dict(coeffs_dict,orient='index',columns=['Beta'])
linear_fit['z_score'] = z_dict.values()

# A z-score with absolute value greater than 2 indicates a significant feature,
# I only see small z scores for sex, and some of the dummy variables
# Since dummy variables treat a single feature as multiple dimensions in the
# observation, we need to use a different significance test (F statistic)
# which will be done in another post.

print(linear_fit)

# TODO F statistic (for dummy vars)
