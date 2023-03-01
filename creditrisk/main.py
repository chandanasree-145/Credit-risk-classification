# %% #* Import all libraries

#Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl
import networkx as nx
import plotly.express as ex
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import f_regression
from sklearn.metrics import r2_score

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pickle

# %% #* Read the data

df_main = pd.read_csv('credit_risk_dataset.csv')
df_main

# %% #* Check nan values, unique values, duplicates and datatypes of the dataframe

def check_columns(df):
    
    column = []
    shape = []
    datatype = []
    unique_values = []
    null_values = []
    nulls_count =[]
    nunique = []

    col_check = pd.DataFrame()
    
    for col_name in df.columns:
        column.append(col_name)
        shape.append(df[col_name].shape)
        datatype.append(df[col_name].dtype)
        unique_values.append(df[col_name].is_unique)
        null_values.append(df[col_name].isnull().any())
        nulls_count.append(df[col_name].isna().sum())
        nunique.append(df[col_name].nunique())
      
    
    col_check['column'] = column
    col_check['shape'] = shape
    col_check['datatype'] = datatype
    col_check['unique_values'] = unique_values
    col_check['null_values'] = null_values
    col_check['null_count'] = nulls_count
    col_check['nunique'] = nunique
    
    return col_check 

check_columns(df_main)

# %% #* Change the datatypes 

df_main.columns

df_main['person_home_ownership'] = df_main['person_home_ownership'].astype('category')
df_main['loan_grade'] = df_main['loan_grade'].astype('category')
df_main['loan_intent'] = df_main['loan_intent'].astype('category')
df_main['cb_person_default_on_file'] = df_main['cb_person_default_on_file'].astype('category')

df_main.dtypes


# %% #* Value counts of every column

for i in df_main.columns:
    print('Column value counts:' , df_main[i].value_counts())



# %% #* Plot scatter plots

#cols= ['person_income', 'person_age', 'person_emp_length', 'loan_amnt', 'cb_person_cred_hist_length', 'loan_percent_income', 'loan_int_rate']

cols = df_main.columns

# scatter plots for numerical data
fig, axs = plt.subplots(nrows =4, ncols = 3, figsize = (20,15))

for i, col_name in enumerate(cols):
    row = i//12
    col = i%12
    axs = axs.T.flatten()
    axs[col].scatter(df_main[col_name], df_main['loan_status'], alpha = 0.4)
    axs[col].set_xlabel(col_name)
    axs[col].set_ylabel('Loan status')
    axs[col].tick_params(axis='x', labelrotation=15)
plt.show()

#fig.savefig('scatter_plots')

# %% #* Check on inconsistensies

df_main.shape
#There are Outliers and null value in emp_length column 
df_main['person_emp_length'].isnull().any()

# remove the rows with outliers
df_main[df_main['person_emp_length']==123.0]
df_main = df_main.drop(labels=[0,210], axis=0)
df_main 

#Check the mean value and fill the nan values with mean value
df_main[df_main['person_emp_length'].isnull()]
df_main['person_emp_length'].describe()
df_main['person_emp_length'].fillna(df_main['person_emp_length'].mean(), inplace=True)

#There are no nan values in emp length

#Now, only loan_int_rate column has null values and does not have any outliers. So directly fill nan values with mean
df_main['loan_int_rate'].isnull().any()
df_main[df_main['loan_int_rate'].isnull()]
df_main['loan_int_rate'].describe()
df_main['loan_int_rate'].fillna(df_main['loan_int_rate'].mean(), inplace=True)

#Person_age > 100 is a rare case so dropping these rows considering them as errors
df_main[df_main['person_age'].between(100,np.inf)]

df_main = df_main[df_main['person_age']<100]
df_main

#Finally no null values in dataframe
check_columns(df_main)

#Check duplicates
df_main[df_main.duplicated(keep=False)]

#Drop duplicates
df_main.drop_duplicates(inplace=True)
df_main.reset_index(drop=True, inplace=True)

df_main
# %% #* Check which numerical values are correlated more
corr = df_main.corr()
plt.figure(figsize=(35,25))
sns.heatmap(corr, annot=True, cmap='coolwarm')
#plt.savefig("correlation.png")

#None of the columns are correlated more than 0.5 with each other 
# %% #* Plot distributions for categorical columns

df_main.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T
df_main.describe(include = "category").T

columns = ["loan_intent", "person_home_ownership", 'person_age']
for col in columns:
    plt.figure(figsize=(9,9))
    sns.countplot(x=df_main["loan_status"], data=df_main, hue=df_main[col])

#pie plot for categorical columns
column = ["loan_intent", "person_home_ownership", 'loan_grade','cb_person_default_on_file']

for col in column:
    fig = ex.pie(df_main, names=col,hole=0.33)
    fig.show()

# %% #* Distribution plots of numerical plots


#cols = ['person_age', 'person_income','loan_amnt']
numerical_cols= [col_name for col_name in df_main.columns if df_main[col_name].dtype in ['int64', 'float64']]
numerical_cols.remove("loan_status")

plt.figure(figsize = (25, 25))
for i in enumerate(numerical_cols):
    #print("the i:", i)
    #print("the i0:", i[0])
    #print("the i1:", i[1])
    plt.subplot(3, 3,i[0]+1)
    #sns.countplot(i[1],data = df)
    sns.distplot(df_main, x = df_main[i[1]])
    plt.title(i[1])

#Count plot for loan status
sns.countplot(x=df_main['loan_status'], data= df_main)
ex.pie(df_main, names='loan_status',hole=0.33)
plt.show()

#%% #! Handling the skewed data

#Check skewness in data
df_main.skew()

#Check skewness in data
df_main.kurtosis()

#Apply log transformations to distribute the data equally
not_skew = np.log(df_main['person_income'])
not_skew.skew()

# Log transform multiple columns in dataframe 
df = df[['col1', 'col2']].apply(lambda x: np.log(x))

#******************Important Observations*************************
      
#Range compared to Interquartile Range is very large. 
# Data is mostly distributed between 23 and 30, and it is right skewed.
# Skewness is used to measure how much the data is skewed. 
# If skew is zero it means there is no skew. However, if it is positive it means it right skewed. 
# In the case of person_age variable, it is 2.58. 
# Kurtosis is used to measure how data is tailed compared to normal distribution. 
# If it equals to 3 it means it is very close to normal distribution. 
# If it is greater than 3, it means that it is very long tailed and has lots of outliers. 
# Kurtosis of "person_age" variable is 18.56, greater than 3. 
# The data is long tailed and potentially has outliers
# Skewness in target variable ??: Use undersampling, oversampling or SMOTE
# However, tree based models are not affected.
#******************Possibile solutions*************************
#1.Log transformations
#2.Remove outliers
#3.Normalize (min-max)
#4.Cube root: when values are too large. Can be applied on negative values
#5.Square root: applied only to positive values
#6.Reciprocal
#7.Square: apply on left skew
#8.Box Cox transformation


# %% #! Create dummies in the dataframe(not required)
columns = ['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']

final_dummies = pd.get_dummies(df_main[columns])

final_dataframe = pd.concat([final_dummies, df_main[['person_age', 'person_income', 'person_emp_length',
'loan_amnt','loan_int_rate', 'loan_percent_income','cb_person_cred_hist_length', 'loan_status']]], axis=1)

final_dataframe

# %% #* Split data 

X = df_main.drop('loan_status', axis=1)
y = df_main['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=1121218
)

#%% #*SK learn pipelines for onehot encoding 
'''
# Create two pipeline one for categories columns and another for numerical columns
categorical_pipeline = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)

numeric_pipeline = Pipeline(
    steps=[("impute", SimpleImputer(strategy="mean")), 
           ("scale", StandardScaler())]
)
'''

# For categories columns we will impute the missing values with the mode of the column and encode them with One-Hot encoding
categorical_pipeline = Pipeline(
    steps=[
        ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ]
)

# For numerical columns we will impute the missing values with the mean of the column and encode them with One-Hot encoding
numeric_pipeline = Pipeline(
    steps=[("scale", StandardScaler())]
)

#Finally, we will combine the two pipelines with a column transformer. 
# To specify which columns the pipelines are designed for, we should first isolate the categorical and numeric feature names:
cat_cols = X_train.select_dtypes(exclude="number").columns
cat_cols
num_cols = X_train.select_dtypes(include="number").columns
num_cols
#Next, we will input these along with their corresponding pipelines into a ColumnTransFormer instance:

full_processor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, num_cols),
        ("categorical", categorical_pipeline, cat_cols),
    ]
)

#%% #* XGB classifier fit

#Before we train the classifier, let’s preprocess the data and divide it into train and test sets:

# Init classifier
xgb_model = XGBClassifier(learning_rate = 0.05)

#pipeline
mypipeline = Pipeline(steps = [("preprocessor", full_processor),
                               ('model', xgb_model)
                              ])

#crossvalidation and scoring
scores = cross_val_score(mypipeline, X_train, y_train,
                              cv=5,
                              scoring="accuracy")

print("MAE score:\n", scores.mean())

#%% #* Hyperoptimization
param_grid = {
    "model__learning_rate": np.arange(0.01,0.3,0.08),
    "model__max_depth":np.arange(1,10,1)
}

hyper = GridSearchCV(
    estimator = mypipeline,
    param_grid = param_grid ,
    scoring = "accuracy",
    verbose = 10,
    cv = 5)

# Fit
hyper.fit(X_train,y_train)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(hyper, open(filename, 'wb'))

# %% #* Printing the best score and estimator

print(hyper.best_score_)
print(hyper.best_estimator_)

# Predict
predict = hyper.best_estimator_.predict(X_test)
test_score = accuracy_score(predict,y_test)
test_score*100

# %% #* Checks
predictions = list(predict)
count = {}

for i in predictions:
    if not i in count:
        count[i]=1
    else:
        count[i] += 1

count

y_test.value_counts()

a = 6331
b = 1772
a+b
#dict(y_test)
#X_test
# %% 
