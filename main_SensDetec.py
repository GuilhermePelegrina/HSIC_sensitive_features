"""
This script proposes an approach to detect sensitive features before training

For a complete detail of our proposal, as well as for cite this code, please refer to:
Pelegrina, G. D., Couceiro, M., & Duarte, L. T. (2023). A statistical approach to detect sensitive features in a group fairness setting. arXiv preprint arXiv:2305.06994.

We consider the following datasets in our analysis:
- Adult Income: https://archive.ics.uci.edu/ml/datasets/adult
- COMPAS Recidivism risk: https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing
- Taiwanese Default Credit: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the
predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.
- LSAC: http://www.seaphe.org/databases.php
"""

''' Importing packages '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import mod_hsicDecompFair # Module with all functions

''' Importing dataset '''
# data_imp is the dataset to import (adult, compas, lsac_new, tcred)
# n_samp is the number of random samples to consider when calculating HSIC (set 0 if use all samples)
# X and y are the full dataset; X_hsic and y_hsic are the subsamples to calculate HSIC (if n_samp ~= 0)
data_imp = 'compas'
n_samp = 0
X, y, X_hsic, y_hsic = mod_hsicDecompFair.func_read_data(data_imp, n_samp)
n_samp = X_hsic.shape[0]

''' Parameters '''
n = X.shape[0] # Number of samples in sensitive group A 
n_hsic = X_hsic.shape[0] # Number of samples in sensitive group A
features_names = X_hsic.columns # Features names
features_to_encode = X_hsic.columns[X_hsic.dtypes==object].tolist() # Categorical features names

''' HSIC / NOCCO analysis '''
# Predefined parameters
param_hsic = 'linear' # HSIC kernel parameter (use it for linear kernel)
param_nocco = 10**(-6) # NOCCO regularizer parameter

# HSIC kernel and NOCCO kernel calculations for vector of outcomes 
hsic = mod_hsicDecompFair.func_kernel(y_hsic, param_hsic) # Use it for linear kernel
# hsic = mod_hsicDecompFair.func_kernel_rbf(y_hsic) # Use it for RBF kernel
hsic_norm = hsic @ np.linalg.inv(hsic + n*param_nocco*np.eye(n_samp))

# Matrices and vectors
depend_extend = np.array(()) # NOCCO dependence measures for all features and sub-features
depend_categorical = np.array(()) # NOCCO dependence measures for categorical features
depend_mean = np.zeros((X_hsic.shape[1],)) # Mean NOCCO dependence measures for all features
depend_max = np.zeros((X_hsic.shape[1],)) # Maximum NOCCO dependence measures for all features
features_extend = list() # Features and sub-features
features_extend_categorical = list() # sub-features
features_encoded_max = X_hsic.columns.tolist() # Features and sub-features with maximum NOCCO

for ii in range(X_hsic.shape[1]):
    X_aux = X_hsic.loc[:,X_hsic.columns[ii]].to_frame()
    
    if len(set(features_to_encode).intersection({X_hsic.columns[ii]})) > 0:
        
        ohe = OneHotEncoder(handle_unknown='ignore')
        
        X_trans = np.eye(X_aux.shape[0]) @ ohe.fit_transform(X_aux).astype(float)
        
        features_encoded = ohe.get_feature_names(X_aux.columns)
        
        for jj in range(X_trans.shape[1]):
            a = X_trans[:,jj].reshape(n_samp,1)
            hsic_a = mod_hsicDecompFair.func_kernel(a, param_hsic) # Use it for linear kernel
            # hsic_a = mod_hsicDecompFair.func_kernel_rbf(a) # Use it for RBF kernel
            nocco_a = np.trace((hsic_a @ np.linalg.inv(hsic_a + n*param_nocco*np.eye(n_samp))) @ hsic_norm)
            depend_extend = np.append(depend_extend,nocco_a)
            depend_categorical = np.append(depend_categorical,nocco_a)
            
            if depend_extend[-1] > depend_max[ii]:
                features_encoded_max[ii] = features_encoded[jj]
            
            depend_mean[ii] = depend_mean[ii] + depend_extend[-1]
            depend_max[ii] = np.max([depend_max[ii], depend_extend[-1]])
            
            features_extend.append(features_encoded[jj])
            features_extend_categorical.append(features_encoded[jj])
            
            print(features_encoded[jj])
    else:
        X_trans = np.array(X_aux)
        a = X_trans[:,0].reshape(n_samp,1)
        hsic_a = mod_hsicDecompFair.func_kernel(a, param_hsic) # Use it for linear kernel
        # hsic_a = mod_hsicDecompFair.func_kernel_rbf(a) # Use it for RBF kernel
        depend_extend = np.append(depend_extend,np.trace((hsic_a @ np.linalg.inv(hsic_a + n*param_nocco*np.eye(n_samp))) @ hsic_norm))
        
        depend_mean[ii] = depend_mean[ii] + depend_extend[-1]
        depend_max[ii] = np.max([depend_max[ii], depend_extend[-1]])
        
        features_extend.append({X_hsic.columns[ii]})
    
    depend_mean[ii] = (1/X_trans.shape[1]) * depend_mean[ii]
    
    print(ii)

''' Selecting sensitive categorical attributes '''
depend_max_median = np.median(depend_max) # Meadian of the maximum NOCCO dependence measures for all features
depend_sensible_max = depend_max[depend_max > depend_max_median] # Maximum NOCCO dependence measures for selected sensible features
depend_sensible_mean = depend_mean[depend_max > depend_max_median] # Mean NOCCO dependence measures for selected sensible features
features_sensible = list(set(features_to_encode).intersection(features_names[depend_max > depend_max_median])) # Sensible categorical features
features_sensible_extend = list() # Sensible sub-features

for ii in range(len(features_sensible)):
   features_sensible_extend.extend([s for s in features_extend_categorical if "{}_".format(features_sensible[ii]) in s])

features_sensible_extend = list(set(features_sensible_extend).intersection(features_extend_categorical))
indices_sensibles = [features_extend_categorical.index(x) for x in features_sensible_extend]
depend_sensible_extend = depend_categorical[indices_sensibles] # NOCCO dependence measures for selected sensible sub-features

# Ordering sensible features (alphabetic order)
indices_sort = sorted(range(len(features_sensible_extend)), key=lambda k: features_sensible_extend[k])
features_sensible = sorted(features_sensible)
features_sensible_extend = sorted(features_sensible_extend)
depend_sensible_extend = depend_sensible_extend[indices_sort]

# Defining the sub-features with the highest dependence degrees
depend_categorical_selec = np.array(())
features_categorical_selec = list()
for ii in range(len(features_encoded_max)):
    if features_extend_categorical.count(features_encoded_max[ii]) > 0:
        index = features_extend_categorical.index(features_encoded_max[ii])
        depend_categorical_selec = np.append(depend_categorical_selec,depend_categorical[index])
        features_categorical_selec.append(features_encoded_max[ii])


''' Evaluating fairness in sensitive categorical attributes '''
nFold = 10 # Defining the number of folds
k_fold = KFold(nFold,shuffle=True) # Defining the k folds (with shuffle)
seed=50
col_trans = make_column_transformer(
                        (OneHotEncoder(),features_to_encode),
                        remainder = "passthrough"
                        )
rf_classifier = RandomForestClassifier(
                      min_samples_leaf=50,
                      n_estimators=150,
                      bootstrap=True,
                      oob_score=True,
                      n_jobs=-1,
                      random_state=seed,
                      max_features='auto')

pipe = make_pipeline(col_trans, rf_classifier)

# 
tp_g_all, tp_gout_all = np.zeros((nFold,len(depend_categorical))), np.zeros((nFold,len(depend_categorical)))
tn_g_all, tn_gout_all = np.zeros((nFold,len(depend_categorical))), np.zeros((nFold,len(depend_categorical)))
fp_g_all, fp_gout_all = np.zeros((nFold,len(depend_categorical))), np.zeros((nFold,len(depend_categorical)))
fn_g_all, fn_gout_all = np.zeros((nFold,len(depend_categorical))), np.zeros((nFold,len(depend_categorical)))

tp, tp_g, tp_gout = np.zeros((nFold,)), np.zeros((nFold,len(depend_sensible_extend))), np.zeros((nFold,len(depend_sensible_extend)))
tn, tn_g, tn_gout = np.zeros((nFold,)), np.zeros((nFold,len(depend_sensible_extend))), np.zeros((nFold,len(depend_sensible_extend)))
fp, fp_g, fp_gout = np.zeros((nFold,)), np.zeros((nFold,len(depend_sensible_extend))), np.zeros((nFold,len(depend_sensible_extend)))
fn, fn_g, fn_gout = np.zeros((nFold,)), np.zeros((nFold,len(depend_sensible_extend))), np.zeros((nFold,len(depend_sensible_extend)))


tp_g_max, tp_gout_max = np.zeros((nFold,len(features_categorical_selec))), np.zeros((nFold,len(features_categorical_selec)))
tn_g_max, tn_gout_max = np.zeros((nFold,len(features_categorical_selec))), np.zeros((nFold,len(features_categorical_selec)))
fp_g_max, fp_gout_max = np.zeros((nFold,len(features_categorical_selec))), np.zeros((nFold,len(features_categorical_selec)))
fn_g_max, fn_gout_max = np.zeros((nFold,len(features_categorical_selec))), np.zeros((nFold,len(features_categorical_selec)))


for kk, (train, test) in enumerate(k_fold.split(X, y, groups=y)):
    
    X_train, X_test, y_train, y_test = X.iloc[train,:], X.iloc[test,:], y.iloc[train], y.iloc[test]
    nSamp_train = X_train.shape[0] # Number of samples (train) and attributes
    nSamp_test = X_test.shape[0] # Number of samples (test)
    
    pipe.fit(X_train, y_train)

    y_class = pipe.predict(X_test)
    
    tn[kk] = sum((y_class < 0) * (y_test < 0))
    tp[kk] = sum((y_class > 0) * (y_test > 0))
    fn[kk] = sum((y_class < 0) * (y_test > 0))
    fp[kk] = sum((y_class > 0) * (y_test < 0))
    
    cont = 0
    for ii in range(len(features_sensible)):
        features_to_encode_aux = [features_sensible[ii]]
        
        X_aux = X_test[features_to_encode_aux]

        ohe = OneHotEncoder(handle_unknown='ignore')
        
        X_trans = np.eye(X_aux.shape[0]) @ ohe.fit_transform(X_aux).astype(float)
       
        for jj in range(X_trans.shape[1]):
            
            tn_g[kk,cont] = sum((y_class < 0).astype(float) * np.array(y_test < 0) * np.squeeze(np.array(X_aux == features_sensible_extend[cont][len(max(features_to_encode_aux, key=len))+1:])))
            tp_g[kk,cont] = sum((y_class > 0).astype(float) * np.array(y_test > 0) * np.squeeze(np.array(X_aux == features_sensible_extend[cont][len(max(features_to_encode_aux, key=len))+1:])))
            fn_g[kk,cont] = sum((y_class < 0).astype(float) * np.array(y_test > 0) * np.squeeze(np.array(X_aux == features_sensible_extend[cont][len(max(features_to_encode_aux, key=len))+1:])))
            fp_g[kk,cont] = sum((y_class > 0).astype(float) * np.array(y_test < 0) * np.squeeze(np.array(X_aux == features_sensible_extend[cont][len(max(features_to_encode_aux, key=len))+1:])))
            
            tn_gout[kk,cont] = sum((y_class < 0).astype(float) * np.array(y_test < 0) * np.squeeze(np.array(X_aux != features_sensible_extend[cont][len(max(features_to_encode_aux, key=len))+1:])))
            tp_gout[kk,cont] = sum((y_class > 0).astype(float) * np.array(y_test > 0) * np.squeeze(np.array(X_aux != features_sensible_extend[cont][len(max(features_to_encode_aux, key=len))+1:])))
            fn_gout[kk,cont] = sum((y_class < 0).astype(float) * np.array(y_test > 0) * np.squeeze(np.array(X_aux != features_sensible_extend[cont][len(max(features_to_encode_aux, key=len))+1:])))
            fp_gout[kk,cont] = sum((y_class > 0).astype(float) * np.array(y_test < 0) * np.squeeze(np.array(X_aux != features_sensible_extend[cont][len(max(features_to_encode_aux, key=len))+1:])))
            
            cont += 1
            
    cont = 0
    for ii in range(len(features_to_encode)):
        features_to_encode_aux = [features_to_encode[ii]]
        
        X_aux = X_test[features_to_encode_aux]

        ohe = OneHotEncoder(handle_unknown='ignore')
        
        X_trans = np.eye(X_aux.shape[0]) @ ohe.fit_transform(X_aux).astype(float)
        
        for jj in range(X_trans.shape[1]):
            
            tn_g_all[kk,cont] = sum((y_class < 0).astype(float) * np.array(y_test < 0) * np.squeeze(np.array(X_aux == features_extend_categorical[cont][len(max(features_to_encode_aux, key=len))+1:])))
            tp_g_all[kk,cont] = sum((y_class > 0).astype(float) * np.array(y_test > 0) * np.squeeze(np.array(X_aux == features_extend_categorical[cont][len(max(features_to_encode_aux, key=len))+1:])))
            fn_g_all[kk,cont] = sum((y_class < 0).astype(float) * np.array(y_test > 0) * np.squeeze(np.array(X_aux == features_extend_categorical[cont][len(max(features_to_encode_aux, key=len))+1:])))
            fp_g_all[kk,cont] = sum((y_class > 0).astype(float) * np.array(y_test < 0) * np.squeeze(np.array(X_aux == features_extend_categorical[cont][len(max(features_to_encode_aux, key=len))+1:])))
            
            tn_gout_all[kk,cont] = sum((y_class < 0).astype(float) * np.array(y_test < 0) * np.squeeze(np.array(X_aux != features_extend_categorical[cont][len(max(features_to_encode_aux, key=len))+1:])))
            tp_gout_all[kk,cont] = sum((y_class > 0).astype(float) * np.array(y_test > 0) * np.squeeze(np.array(X_aux != features_extend_categorical[cont][len(max(features_to_encode_aux, key=len))+1:])))
            fn_gout_all[kk,cont] = sum((y_class < 0).astype(float) * np.array(y_test > 0) * np.squeeze(np.array(X_aux != features_extend_categorical[cont][len(max(features_to_encode_aux, key=len))+1:])))
            fp_gout_all[kk,cont] = sum((y_class > 0).astype(float) * np.array(y_test < 0) * np.squeeze(np.array(X_aux != features_extend_categorical[cont][len(max(features_to_encode_aux, key=len))+1:])))
            
            if features_categorical_selec.count(features_extend_categorical[cont]) > 0:
                index = features_categorical_selec.index(features_extend_categorical[cont])
                tp_g_max[kk,index], tp_gout_max[kk,index] = tp_g_all[kk,cont], tp_gout_all[kk,cont]
                tn_g_max[kk,index], tn_gout_max[kk,index] = tn_g_all[kk,cont], tn_gout_all[kk,cont]
                fp_g_max[kk,index], fp_gout_max[kk,index] = fp_g_all[kk,cont], fp_gout_all[kk,cont]
                fn_g_max[kk,index], fn_gout_max[kk,index] = fn_g_all[kk,cont], fn_gout_all[kk,cont]

            cont += 1
            
        
    
    print(kk)
    
    
tpr, fpr = tp/(tp+fn), fp/(fp+tn)
tpr[np.isnan(tpr)], fpr[np.isnan(fpr)] = 0, 0
tpr_mean, tpr_std, fpr_mean, fpr_std = np.mean(tpr, axis=0), np.std(tpr, axis=0), np.mean(fpr, axis=0), np.std(fpr, axis=0)

accur = (tp + tn)/(tp + tn + fp + fn)
accur[np.isnan(accur)] = 0
accur_mean, accur_std = np.mean(accur, axis=0), np.std(accur, axis=0)

tpr_g_mean, tpr_g_std, fpr_g_mean, fpr_g_std, tpr_gout_mean, tpr_gout_std, fpr_gout_mean, accur_g_mean, accur_g_std, accur_gout_mean, accur_gout_std, eq_op_mean, eq_op_std, pr_pa_mean, pr_pa_std, eq_od_mean, eq_od_std, ov_ac_mean, ov_ac_std, di_im_mean, di_im_std = mod_hsicDecompFair.func_metrics(tp_g, fp_g, tn_g, fn_g, tp_gout, fp_gout, tn_gout, fn_gout)

tpr_g_mean_all, tpr_g_std_all, fpr_g_mean_all, fpr_g_std_all, tpr_gout_mean_all, tpr_gout_std_all, fpr_gout_mean_all, accur_g_mean_all, accur_g_std_all, accur_gout_mean_all, accur_gout_std_all, eq_op_mean_all, eq_op_std_all, pr_pa_mean_all, pr_pa_std_all, eq_od_mean_all, eq_od_std_all, ov_ac_mean_all, ov_ac_std_all, di_im_mean_all, di_im_std_all = mod_hsicDecompFair.func_metrics(tp_g_all, fp_g_all, tn_g_all, fn_g_all, tp_gout_all, fp_gout_all, tn_gout_all, fn_gout_all)

tpr_g_mean_max, tpr_g_std_max, fpr_g_mean_max, fpr_g_std_max, tpr_gout_mean_max, tpr_gout_std_max, fpr_gout_mean_max, accur_g_mean_max, accur_g_std_max, accur_gout_mean_max, accur_gout_std_max, eq_op_mean_max, eq_op_std_max, pr_pa_mean_max, pr_pa_std_max, eq_od_mean_max, eq_od_std_max, ov_ac_mean_max, ov_ac_std_max, di_im_mean_max, di_im_std_max = mod_hsicDecompFair.func_metrics(tp_g_max, fp_g_max, tn_g_max, fn_g_max, tp_gout_max, fp_gout_max, tn_gout_max, fn_gout_max)

''' Plots '''

performance_measures = ['Overal accuracy equality','Equal opportunity','Predictive parity','Equalized odds']

# Dependence measure
plt.show()
mod_hsicDecompFair.func_max_depend(depend_max,features_encoded_max)

# Dependence measure x fairness measures    
plt.show()
mod_hsicDecompFair.func_scatter(features_categorical_selec,depend_categorical_selec,ov_ac_mean_max,eq_op_mean_max,pr_pa_mean_max,eq_od_mean_max,performance_measures)

data_save = [depend_categorical,depend_categorical_selec,depend_max,depend_max_median,depend_mean,depend_sensible_extend,depend_sensible_max,depend_sensible_mean,features_categorical_selec,features_encoded_max,features_extend,features_extend_categorical,features_names,features_sensible,features_sensible_extend,features_to_encode,tp,tp_g,tp_gout,fp,fp_g,fp_gout,tn,tn_g,tn_gout,fn,fn_g,fn_gout,tp_g_all,tp_gout_all,fp_g_all,fp_gout_all,tn_g_all,tn_gout_all,fn_g_all,fn_gout_all,tp_g_max,tp_gout_max,fp_g_max,fp_gout_max,tn_g_max,tn_gout_max,fn_g_max,fn_gout_max]
np.save('results_SensDetec_test.npy', data_save, allow_pickle=True)
#depend_categorical,depend_categorical_selec,depend_max,depend_max_median,depend_mean,depend_sensible_extend,depend_sensible_max,depend_sensible_mean,features_categorical_selec,features_encoded_max,features_extend,features_extend_categorical,features_names,features_sensible,features_sensible_extend,features_to_encode,tp,tp_g,tp_gout,fp,fp_g,fp_gout,tn,tn_g,tn_gout,fn,fn_g,fn_gout,tp_g_all,tp_gout_all,fp_g_all,fp_gout_all,tn_g_all,tn_gout_all,fn_g_all,fn_gout_all,tp_g_max,tp_gout_max,fp_g_max,fp_gout_max,tn_g_max,tn_gout_max,fn_g_max,fn_gout_max = np.load('results_SensDetec_tcred_linear.npy', allow_pickle=True)





