import os #paths to file
import numpy as np # linear algebra
import pandas as pd # data processing
import warnings# warning filter
#ploting libraries
import matplotlib.pyplot as plt 
import seaborn as sns
#relevant ML libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics
#ML models
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#default theme
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)
#warning hadle
warnings.filterwarnings("ignore")
#path for the training set
tr_path = "datastrain.csv"
#path for the testing set
te_path = "datastest.csv"
# read in csv file as a DataFrame
tr_df = pd.read_csv(tr_path)
# explore the first 5 rows
print(tr_df.head())
# read in csv file as a DataFrame
te_df = pd.read_csv(te_path)
# explore the first 5 rows
print(te_df.head())
print(f"training set (row, col): {tr_df.shape}\n\ntesting set (row, col): {te_df.shape}")
#column information
print(tr_df.info(verbose=True, null_counts=True))
#summary statistics
print(tr_df.describe())
#the Id column is not needed, let's drop it for both test and train datasets

#checking the new shapes
print(f"training set (row, col): {tr_df.shape}\n\ntesting set (row, col): {te_df.shape}")
#missing values in decsending order
print(tr_df.isnull().sum().sort_values(ascending=False))
#filling the missing data
print("Before filling missing values\n\n","#"*50,"\n")

#list of all the columns.columns
#Cols = tr_df.tolist()
#list of all the numeric columns
num = tr_df.select_dtypes('number').columns.to_list()
#list of all the categoric columns
cat = tr_df.select_dtypes('object').columns.to_list()

#numeric df
cancer_num =  tr_df[num]
#categoric df
cancer_cat = tr_df[cat]

print(tr_df[cat[-1]].value_counts())
#tr_df[cat[-1]].hist(grid = False)
#print(i)
total = float(len(tr_df[cat[-1]]))
plt.figure(figsize=(8,10))
sns.set(style="whitegrid")
ax = sns.countplot(tr_df[cat[-1]])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}'.format(height/total),ha="center") 
plt.show()
for i in cancer_num:
    plt.hist(cancer_num[i])
    plt.title(i)
    plt.show()
for i in cat[:-1]: 
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    sns.countplot(x=i ,hue='status', data=tr_df ,palette='plasma')
    plt.xlabel(i, fontsize=14)
    
#converting categorical values to numbers

to_numeric = {'Affected': 1, 'Non Affected': 2}

# adding the new numeric values from the to_numeric variable to both datasets
tr_df = tr_df.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)
te_df = te_df.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)


# checking the our manipulated dataset for validation
print(f"training set (row, col): {tr_df.shape}\n\ntesting set (row, col): {te_df.shape}\n")
print(tr_df.info(), "\n\n", te_df.info())

#plotting the correlation matrix
print(sns.heatmap(tr_df.corr() ,cmap='cubehelix_r'))

#correlation table
corr = tr_df.corr()
print(corr.style.background_gradient(cmap='coolwarm').set_precision(2))

y = tr_df['status']
X = tr_df.drop('status', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
#y_test = y_test.map({'True': 1, 'False': 0}).astype(int)


DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
y_predict = DT.predict(X_test)
#  prediction Summary by species
print(classification_report(y_test, y_predict))
# Accuracy score
DT_SC = accuracy_score(y_predict,y_test)
print(f"{round(DT_SC*100,2)}% Accurate")
Decision_Tree=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
Decision_Tree.to_csv("Decision Tree.csv")  


RF = RandomForestClassifier()
RF.fit(X_train, y_train)
y_predict = RF.predict(X_test)
#  prediction Summary by species
print(classification_report(y_test, y_predict))
# Accuracy score
RF_SC = accuracy_score(y_predict,y_test)
print(f"{round(RF_SC*100,2)}% Accurate")
Random_Forest=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
Random_Forest.to_csv("Random Forest.csv")  


XGB = XGBClassifier()
XGB.fit(X_train, y_train)
y_predict = XGB.predict(X_test)
# prediction Summary by species
print(classification_report(y_test, y_predict))
# Accuracy score
XGB_SC = accuracy_score(y_predict,y_test)
print(f"{round(XGB_SC*100,2)}% Accurate")
XGBoost=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
XGBoost.to_csv("XGBoost.csv")   




LR = LogisticRegression()
LR.fit(X_train, y_train)
y_predict = LR.predict(X_test)
#  prediction Summary by species
print(classification_report(y_test, y_predict))
# Accuracy score
LR_SC = accuracy_score(y_predict,y_test)
print('accuracy is',accuracy_score(y_predict,y_test))
Logistic_Regression=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
Logistic_Regression.to_csv("Logistic Regression.csv")   


#create ROC curve
y_pred_proba = LR.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba, pos_label=1)
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print(y_test)

  
score = [DT_SC,RF_SC,XGB_SC,LR_SC]
Models = pd.DataFrame({
    'n_neighbors': ["Decision Tree","Random Forest","XGBoost", "Logistic Regression"],
    'Score': score})
print(Models.sort_values(by='Score', ascending=False))

