import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
from sklearn.decomposition import PCA
import sklearn.feature_selection
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# load dataframe
df = pd.read_csv('D://learning materials//2019 autumn//Fundamentals of Data Analytics//ass3//Assignment3_TrainingSet.csv')

#function for replacing missing values
def rpmissing(df):
    df = df.replace('unknown', np.NaN)
    df = df.fillna(df.mode().iloc[0])
    return df

# function for resampling
# def resample(df):
#     # Class count
#     count_class_0, count_class_1 = df.Final_Y.value_counts()
#     df_class_0 = df[df['Final_Y'] == 0]
#     df_class_1 = df[df['Final_Y'] == 1]
#     df_class_1_over = df_class_1.sample(count_class_0, replace=True)
#     df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)
#     return df_test_over
# function for seperate feacure
# def seperatefeature(df):
#     x = df.drop('Final_Y', 1)
#     x = x.drop('row ID', 1)
#     y = df.Final_Y.astype(int)
#     return x,y

# function for binning values
def binning(x):
    # convert 'education' into numerical values
    education_dict = {'illiterate':1 , 'basic.4y': 2, 'basic.6y': 3, 'basic.9y': 4, 'high.school': 5, 'university.degree': 6, 'professional.course': 7}
    x.education.replace(education_dict,inplace=True)
    # grouping month values
    x['month']=['high.season' if m in['may','jul','aug','jun','nov','apr'] else 'off.season' for m in x['month']]
    # grouping job values
    x['job']=['senior.level' if j in['management','entrepreneur'] else 'junior.level'if j in ['admin.','blue-collar','technician','services'] else 'others' for j in x['job']]
    #grouping pdays values
    x['pdays']=['offlimits' if c == 999 else 'recent' for c in x.pdays]
    return x
# function to dummy all the categorical variables and numerical attributes with only two classes
def dummy_df(df):
    todummy_list=[]
    for col_name in df.columns:
        if df[col_name].dtypes == 'object'or len(df[col_name].unique()) == 2:
            todummy_list.append(col_name)
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df=df.drop(x,1)
        df=pd.concat([df,dummies], axis=1)
    return df
# function to normalize all numerical attributes more that two classes
def normalize(df):
    for col_name in df.columns:
        if len(df[col_name].unique()) > 2 and df[col_name].dtypes != 'object':
            df[col_name]=stats.zscore(df[col_name])
    return df
# Two way Interactions amongst features
def add_interaction(df):
    # et feature names
    combos =list(combinations(list(df.columns),2))
    colnames=list(df.columns)+[''.join(x) for x in combos]
    # find interactions
    poly = PolynomialFeatures(interaction_only=True,include_bias=False)
    df=poly.fit_transform(df)
    df=pd.DataFrame(df)
    df.columns=colnames
    # Remve interaction terms with all 0 values
    noint_indicies = [i for i, x in enumerate(list((df == 0).all())) if x]
    df=df.drop(df.columns[noint_indicies],axis=1)
    return df

# Dinebsuibakty reduction using PCA
def conductPCA(df, n):
    pca = PCA(n_components=n)
    x_pca=pd.DataFrame(pca.fit_transform(df))
    return x_pca

# Feature selection
def featureselec(x,x_train,y_train,n):
    select =sklearn.feature_selection.SelectKBest(k=n)
    selected_features = select.fit(x_train, y_train)
    indices_selected = selected_features.get_support(indices=True)
    colnames_selected=[x.columns[i] for i in indices_selected]
    return colnames_selected


# define preprocessing method
def preprocessing (df, istrain):
    df=rpmissing(df)
    if istrain==True:
        count_class_0, count_class_1 = df.Final_Y.value_counts()
        df_class_0 = df[df['Final_Y'] == 0]
        df_class_1 = df[df['Final_Y'] == 1]
        df_class_1_over = df_class_1.sample(count_class_0, replace=True)
        df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)
        x = df_test_over.drop('Final_Y',1)
        x = x.drop('row ID',1)
        y = df_test_over.Final_Y.astype('category',copy=False)
    else:
        x = df.drop('row ID',1)
    df=binning(x)
    df=dummy_df(df)
    # df=normalize(df)
    if istrain == True:
        return df,y
    else:
        return df

# preprocess trainningseet
x,y = preprocessing(df, True)

# data split
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=1, test_size=0.2)





# load test dataset


# aggreate models
# get KNeighborsRegressor result
bestKNN=  KNeighborsRegressor(n_neighbors=2)
bestKNN.fit(x_train,y_train)
KNNrs =  bestKNN.predict(x_test)
KNNrs = [1 if x >= 0.6 else 0 for x in KNNrs]
print("KNNRG results has ",len(KNNrs),"values")
# get KNeighborsClassifier result
bestKNN_CF=  KNeighborsClassifier(n_neighbors=2)
bestKNN_CF.fit(x_train,y_train)
KNNcfrs =  bestKNN_CF.predict(x_test)
print("KNNCF results has ",len(KNNcfrs),"values")

# get RandomForestRegressor result
bestRF = RandomForestRegressor(n_estimators = 396)
bestRF.fit(x_train,y_train)
RFrs =  bestRF.predict(x_test)
RFrs = [1 if x >= 0.9 else 0 for x in RFrs]
print("RFRG results has ",len(RFrs),"values")

# get RF_Classifier result
bestRFCF = RandomForestClassifier(n_estimators = 192)
bestRFCF.fit(x_train,y_train)
RFCFrs =  bestRFCF.predict(x_test)
print("RFCF results has ",len(RFCFrs),"values")
# get ANN result
bestANN = MLPClassifier(hidden_layer_sizes=(7, 7, 7), max_iter=340)
bestANN.fit(x_train,y_train)
ANNrs =  bestANN.predict(x_test)
print("ANN results has ",len(ANNrs),"values")
# get SVM result
bestSVM =  svm.SVC(gamma='auto_deprecated')
bestSVM.fit(x_train,y_train)
SVMrs =  bestSVM.predict(x_test)
print("SVM results has ",len(SVMrs),"values")
# get LR result
bestLR =LogisticRegression(random_state=0, solver='sag',multi_class='multinomial')
bestLR.fit(x_train,y_train)
LRrs =  bestLR.predict(x_test)
LRrs = [1 if x >= 0.1 else 0 for x in LRrs]
print("LR results has ",len(LRrs),"values")
combo = np.sum([KNNrs,RFrs,ANNrs,SVMrs,LRrs], axis = 0)

AGres = [1 if x > 2 else 0 for x in combo]
cm=confusion_matrix(y_test, AGres)
print("The confusion_matrix for the best AggregateClassifier is :")
print(cm)
print("with f1_score = ",f1_score(y_test,AGres))
print("accuracy_score = ",accuracy_score(y_test,AGres))
print("AUC = ",roc_auc_score(y_test, AGres))




# # testing model
# kaggle
test = pd.read_csv('D://learning materials//2019 autumn//Fundamentals of Data Analytics//ass3//Assignment3_TestingSet.csv')
preprocessedtest =preprocessing (test, False)
print (preprocessedtest.head(5))


# Combine
Test_rs_KNNRG =  bestKNN.predict(preprocessedtest)
Test_rs_KNNRG = [1 if x >= 0.6 else 0 for x in Test_rs_KNNRG]
Test_rs_KNNCF =  bestKNN_CF.predict(preprocessedtest)
Test_rs_RFRG =  bestRF.predict(preprocessedtest)
Test_rs_RFRG = [1 if x >= 0.9 else 0 for x in Test_rs_RFRG]
Test_rs_RFCF =  bestRFCF.predict(preprocessedtest)
Test_rs_ANN =  bestANN.predict(preprocessedtest)
Test_rs_SVM =  bestSVM.predict(preprocessedtest)
Test_rs_LR =  bestLR.predict(preprocessedtest)

test_combo = np.sum([Test_rs_KNNRG,Test_rs_KNNCF,Test_rs_RFRG,Test_rs_RFCF,Test_rs_ANN,Test_rs_SVM,Test_rs_LR], axis = 0)
test_res = [1 if x > 3 else 0 for x in test_combo]

output= pd.DataFrame(test_res)
output.columns = ['Final_Y']
output.index= output.index+1
output.to_csv('Aggregate_output.csv',index_label='row ID')