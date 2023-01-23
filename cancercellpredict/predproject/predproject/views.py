from django.shortcuts import render, redirect


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

def home(request):
    return render(request, 'home.html')
def predict(request):
    return render(request, 'predict.html')

def result(request):
    sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)
    warnings.filterwarnings("ignore")
    tr_path = "C:\\Users\\anany\Desktop\VIT\semester5\python lab\Same project\datastrain.csv"
    te_path = "C:\\Users\\anany\Desktop\VIT\semester5\python lab\Same project\datastest.csv"
    tr_df = pd.read_csv(tr_path)
    te_df = pd.read_csv(te_path)
    tr_df.isnull().sum().sort_values(ascending=False)
    num = tr_df.select_dtypes('number').columns.to_list()
    cat = tr_df.select_dtypes('object').columns.to_list()
    cancer_num =  tr_df[num]
    cancer_cat = tr_df[cat]
    total = float(len(tr_df[cat[-1]]))
    to_numeric = {'Affected': 1, 'Non Affected': 2}
    tr_df = tr_df.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)
    te_df = te_df.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)
    corr = tr_df.corr()
    y = tr_df['status']
    X = tr_df.drop('status', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    LR = LogisticRegression()
    LR.fit(X_train, y_train)
    
    val1=float(request.GET['WBC_area'])
    val2=float(request.GET['WBC_convex_area'])
    val3=float(request.GET['WBC_peri'])
    val4=float(request.GET['ecc_wbc'])
    val5=float(request.GET['solidity_wbc'])
    val6=float(request.GET['orient_wbc'])
    val7=float(request.GET['nuc_area'])
    val8=float(request.GET['nuc_ratio'])
    val9=float(request.GET['peri_nuc'])
    val10=float(request.GET['round_nuc'])
    val11=float(request.GET['ecc_nuc'])
    val12=float(request.GET['solidity_nuc'])
    val13=float(request.GET['convex_area_nuc'])
    val14=float(request.GET['avg_cyt_re'])
    val15=float(request.GET['avg_cyt_gr'])
    val16=float(request.GET['avg_cyt_bl'])
    val17=float(request.GET['entropy_cyt'])
    val18=float(request.GET['minoraxis'])
    val19=float(request.GET['majoraxis'])
    val20=float(request.GET['minoraxis_nuc'])
    val21=float(request.GET['majoraxis_nuc'])
    val22=float(request.GET['axismeanratio'])
    
    y_predict=LR.predict([[val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14, val15, val16, val17, val18, val19, val20, val21, val22]])
       
    result1=""
    if y_predict==[1]:
        result1="Affected"
    else:
        result1="Not Affected"
        
    
    #create ROC curve
    y_pred_proba = LR.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba, pos_label=1)
    fig = plt.figure(figsize=(2,2))
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    fig.savefig('C:\\Users\\anany\Desktop\VIT\semester5\python lab\Same project\predproject\static\\rocplot.jpg', bbox_inches='tight', dpi=150)
    

    
    return render(request, 'predict.html', {"result2":result1})












