# Exported script of excercise notebook
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pickle
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_auc_score
from sklearn.metrics import roc_curve
from IPython import get_ipython

# %% [markdown]
# # Evaluation metrics
#
# Overview of different evaluation metrics that can be used with different models.

# %%
# import necessary dependencies:
import pandas as pd
import numpy as np

# Formatting output display
from IPython.display import display

# Plotting
import plotly.graph_objs as go
import plotly.offline as py

# Data validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score

# One hot encoding
from sklearn.feature_extraction import DictVectorizer

# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Accuracy
from sklearn.metrics import accuracy_score


# %%
# Dataset details - saved directly from kaggle
df = pd.read_csv('churn_data.csv')
df.head(10)

# %% [markdown]
# ## Data cleaning
#
# Clean and preprocess the data

# %%
# Preprocess the column names - all lowercase
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns


# %%
# Process the string columns:
columns_with_strings = list(df.dtypes[df.dtypes == 'object'].index)

# Correct all the lower case:
for column in columns_with_strings:
    df[column] = df[column].str.lower().str.replace(' ', '_')

df.tail(10)


# %%
# Total charges column is of string type, bu should be numeric
total_charges = pd.to_numeric(df['totalcharges'], errors='coerce')

# Check corresponding customer ids for which totalcharges are null
df[total_charges.isnull()][['customerid', 'totalcharges']]


# %%
# Fill the values using zerofill
df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')

df['totalcharges'] = df['totalcharges'].fillna(0)

# %% [markdown]
# ## Processing categorical data
#

# %%
# Converting the churn into binary - 0 for no, 1 otherwise.
df['churn'] = df['churn'].apply(lambda val: val == 'yes').astype(int)
df['churn'].head(10)

# %% [markdown]
# ## Validation framework
#
# Validation framework setup using `scikit-learn`

# %%
# Set up test data
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

print("Length of the training set (sans validation set): {}\nLength of the test set: {}".format(
    len(df_full_train), len(df_test)))


# %%
# Set up validation data
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=1)
print("Length of training data: {}\nLength of validation data: {}".format(
    len(df_train), len(df_val)))
df_train.head(10)

# %% [markdown]
# `test_size` was set to `0.25` in the previous cell as the split was done on `full_train`, **not on the original dataset**.
# 20% of 80% = 25%

# %%
# Reset indices:


def split_data(data):
    """
    Helper function for:
    1. Resetting index of the dataframe - For code readability
    2. Split the input from output
    """
    data = data.reset_index(drop=True)
    # Separate the output
    output = data['churn'].values
    # delete the columns
    del data['churn']
    return data, output


# %%
# Reset index and split input from output

df_train, y_train = split_data(df_train)
df_val, y_val = split_data(df_val)
df_test, y_test = split_data(df_test)
# df_train.head(10)

# %% [markdown]
# ## Exploratory data analysis
#
# * Handle missing values
# * Examine the output column
# * Process categorical data

# %%
df_full_train = df_full_train.reset_index(drop=True)
# df_full_train['churn'].value_counts(normalize=True)


# %%
# assign columns that are categorical
categorical_columns = [
    column for column in df_full_train.columns if df_full_train[column].nunique() <= 5]

# Remove the output column
categorical_columns.remove('churn')
categorical_columns


# %%
# Numeric columns
numeric_columns = [
    column for column in df_full_train.columns if df_full_train[column].dtype != 'object']

# Drop output column
numeric_columns.remove('churn')

numeric_columns

# %% [markdown]
# ### Numerical data
#
# **Mutual Information**
#
# Amount of information commmon between two variables - deals with entropy of a variable. Useful for examinimg categorical data.
#
# **Correlation**
# Relation between two variables - useful for examining numerical data

# %%
# Correlation:
numeric_columns.append('churn')

# Generate correlation matrix
numeric_data = df_full_train[numeric_columns]
correlation_data = numeric_data.corr()
corr_matrix = correlation_data.values


# %%
# Set up plotting environment

# Text info to display the correlation information
text_info = np.round(corr_matrix, decimals=2).astype(str)

# Layout
Layout = go.Layout(title='Correlation heatmap of numerical data', autosize=False, width=600,
                   height=600)

# Data
Data = [go.Heatmap(x=numeric_columns, y=numeric_columns,
                   z=corr_matrix, text=text_info)]

figure = go.Figure(data=Data, layout=Layout)

py.iplot(figure)


# %%
# Mutual information
def mutual_info(parameter):
    """
    Returns the mutual information score between categorical column and output.
    In this case - output = df_full_train['churn']
    """
    return mutual_info_score(parameter, df_full_train['churn'])


m_score = df_full_train[categorical_columns].apply(mutual_info)
m_score.sort_values()

# %% [markdown]
# ## One hot encoding
#
# Encode all categorical columns using:
# * Dictvectorizer
# * Onehoencoder
# of `scikit-learn`
#
#

# %%
# Bug fix
numeric_columns.remove('churn')

# Variation 1: use dictvectorizer:
train_dicts = df_train[categorical_columns +
                       numeric_columns].to_dict(orient='records')

# Initialize the vectorizer:
dv = DictVectorizer()

# Encode the training data
x_train = dv.fit_transform(train_dicts)


# %%
# Encode the validation data
val_dicts = df_val[categorical_columns +
                   numeric_columns].to_dict(orient='records')

# Encode the validation data
x_val = dv.transform(val_dicts)

# %% [markdown]
# ## Logistic Regression
#
# Training the model using logistic regression

# %%
# Initializing model
model = LogisticRegression()
model.fit(x_train, y_train)


# %%
# Test and verify predictions
y_pred_val = model.predict_proba(x_val)[:, 1]

# Taking only those whose possibility of churn is greater than 0.5
churn_decision = (y_pred_val >= 0.5)

# Compare results:
df_predictions = pd.DataFrame()
df_predictions['predicted_usin_probs'] = churn_decision.astype(int)
df_predictions['predictions'] = model.predict(x_val)
df_predictions['actual'] = y_val

# Estimating accuracy:
df_predictions['correct'] = (
    df_predictions['predictions'] == df_predictions['actual'])

df_predictions


# %%
# Encode the test data
test_dicts = df_test[categorical_columns +
                     numeric_columns].to_dict(orient='records')

x_test = dv.transform(test_dicts)

# Making predictions
# Making predictions
y_test_predict = model.predict_proba(x_test)[:, 1]
churn_decision_test = (y_test_predict >= 0.5)

# Checking the accuracy of the test
(churn_decision_test == y_test).mean()

# %% [markdown]
# ## Evaluation metrics
#
# Accuracy measured at a threshold

# %%
# Generating the thresholds
thresholds = np.linspace(0, 1, 21)

scores = []

# Append the scores
for t in thresholds:
    score = accuracy_score(y_val, y_pred_val >= t)
    print('%.2f %.3f' % (t, score))
    scores.append(score)

scores


# %%
# Plot the data
data = [go.Scatter(x=thresholds, y=scores, mode='lines')]

layout = go.Layout(title='Accuracy at various thresholds for churn',
                   xaxis_title='Threshold', yaxis_title='Accuracy')

figure = go.Figure(data=data, layout=layout)

py.iplot(figure)

# %% [markdown]
# Reason why the plot above is specified as accuracy for churn is that it shows the accuracy for only one class of outcome (recall the definition of logistic regression).
#
# For the other class (not churn)

# %%
# Accuracy for the other class on validation data
1 - y_val.mean()

# %% [markdown]
# ## Confusion matrix
#
# Used for classification algorithms -

# %%
# Confusion matrix from scratch
# Setting true and false
true = (y_val == 1)
false = (y_val == 0)

# Setting the prediction
pred_positive = (y_pred_val >= 0.5)
pred_negative = (y_pred_val < 0.5)

# The confusion matrix
tp = (pred_positive & true).sum()
tn = (pred_negative & false).sum()
fp = (pred_positive & false).sum()
fn = (pred_negative & true).sum()

confusion_matrix = np.array([[tn, fp], [fn, tp]])
print(confusion_matrix)

# Percentage of values
confusion_matrix / confusion_matrix.sum()

# %% [markdown]
# ## Precision and recall
#
# $$
# \textup{Precision}=\frac{\textup{TP}}{\textup{TP}+\textup{FP}} \\
# \textup{Recall} = \frac{\textup{TP}}{\textup{TP}+\textup{FN}}
# $$
#
#

# %%
# Calculating precision and recall
precision = tp / (tp+fp)
recall = tp/(tp+fn)

print("Precision: {}\nRecall: {}".format(precision, recall))


# %%
# Calculating for all thresholds
thresholds = np.linspace(0, 1, 101)

confusion_score = []

for t in thresholds:
    pred_positive = (y_pred_val >= t)
    pred_negative = (y_pred_val < t)

    # The confusion matrix
    tp = (pred_positive & true).sum()
    tn = (pred_negative & false).sum()
    fp = (pred_positive & false).sum()
    fn = (pred_negative & true).sum()

    # Add the scores
    confusion_score.append([t, tp, fp, fn, tn])

# Convert to dataframe
confusion_scores = pd.DataFrame(confusion_score, columns=['threshold', 'true_positive',
                                                          'false_positive', 'false_negative', 'true_negative'])

confusion_scores.head()


# %%
# Adding precision and recall
confusion_scores['tpr'] = confusion_scores['true_positive'] / \
    (confusion_scores['true_positive'] + confusion_scores['false_negative'])
confusion_scores['fpn'] = confusion_scores['false_positive'] / \
    (confusion_scores['false_positive'] + confusion_scores['true_negative'])

confusion_scores.head()


# %%
# Plotting the true positive and false positive rates against threshold
trace1 = go.Scatter(x=confusion_scores['threshold'],
                    y=confusion_scores['tpr'], mode='lines', name='tpr')
trace2 = go.Scatter(x=confusion_scores['threshold'],
                    y=confusion_scores['fpn'], mode='lines', name='fpr')

data = [trace1, trace2]

layout = go.Layout(title='Scores plot against threshold',
                   xaxis_title='Threshold', yaxis_title='Positive rates')

figure = go.Figure(data=data, layout=layout)

py.iplot(figure)

# %% [markdown]
# ## ROC Curve

# %%
# ROC Curve from scratch
trace1 = go.Scatter(x=[0, 1], y=[0, 1], mode='lines+markers',
                    line=dict(dash='dash'), name='Reference')
trace2 = go.Scatter(
    x=confusion_scores['fpn'], y=confusion_scores['tpr'], name='Model')

data = [trace1, trace2]

layout = go.Layout(title='ROC Curve', xaxis_title='False positive rates',
                   yaxis_title='True positive rates', width=600, height=500)

figure = go.Figure(data=data, layout=layout)

py.iplot(figure)


# %%
# Verifying with sklearn

fpr, tpr, thresholds = roc_curve(y_val, y_test)

# Plot the curve
race1 = go.Scatter(x=[0, 1], y=[0, 1], mode='lines+markers',
                   line=dict(dash='dash'), name='Reference')
trace2 = go.Scatter(
    x=confusion_scores['fpn'], y=confusion_scores['tpr'], name='Model')

data = [trace1, trace2]

layout = go.Layout(title='ROC Curve', xaxis_title='False positive rates',
                   yaxis_title='True positive rates', width=600, height=500)

figure = go.Figure(data=data, layout=layout)

py.iplot(figure)


# %%
# AUC score

print(auc(tpr, fpr))
print(roc_auc_score(y_val, y_pred_val))

# %% [markdown]
# ## K-Fold cross validation

# %%
# numeric_columns.remove('churn')
numeric_columns


# %%
# Defining the training module
def train(train_data, target, C):
    """
    Helper function for training
    """
    dicts = train_data[categorical_columns +
                       numeric_columns].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    x_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C)
    model.fit(x_train, y_train)

    return dv, model


# %%
# Prediction functions
def predict(test_data, dv, model):
    """
    Predict values
    """
    dicts = test_data[categorical_columns +
                      numeric_columns].to_dict(orient='records')
    x_test = dv.transform(dicts)
    prediction = model.predict_proba(x_test)[:, 1]

    return prediction


# %%
# Sample model - will be used for deploying
dv, model_to_deploy = train(df_train, y_train, C=1)
test_pred = predict(df_test, dv, model_to_deploy)

auc_score = roc_auc_score(y_test, test_pred)
auc_score

# %% [markdown]
# ## Regularization for the model
#
# It can be noted from the k-fold scores to decide which value would be best

# %%
get_ipython().system('pip install tqdm')


# %%
# Validation

# Displaying the proces


# %%
# Choosing the regularization value
reg_value = [0.001, 0.01, 0.1, 0.5, 1, 5, 10]

# Appending the scores


# No. of splits
n_splits = 5

# K-Fold validation
for value in tqdm(reg_value):
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    scores = []

    # Validation for 1 split
    for train_idx, val_idx in k_fold.split(df_full_train):

        # Split into training and validation
        train_set = df_full_train.iloc[train_idx]
        val_set = df_full_train.iloc[val_idx]

        # Seperate the target and input
        y_train = train_set.churn.values
        y_val = val_set.churn.values

        # Train the model
        dv, model = train(train_set, y_train, C=value)

        # Make predictions
        y_pred = predict(val_set, dv, model)

        # Calculate auc scores
        auc = roc_auc_score(y_val, y_pred)

        # Append the scores
        scores.append(auc)

    # Give the score for 1 val
    print("C = {}: score = {:.2f}±{:.2f}".format(
        value, np.mean(scores), np.std(scores)))

# %% [markdown]
# ## Saving the model

# %%


# %%
# Prepare the output file
output_file = f"model_C={1}.bin"

output_file


# %%
# Write the contents (binary)

# Write the contents - the with statement opens and closes the file (no need for additional commands)
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model_to_deploy), f_out)

# %% [markdown]
# ## Load the model

# %%
# Open the file in reading mode
with open(output_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


# %%
# Use the model:
new_customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


# %%
x = dv.transform([new_customer])


# %%
# Predict churn
print(model.predict_proba(x)[0, 1])
