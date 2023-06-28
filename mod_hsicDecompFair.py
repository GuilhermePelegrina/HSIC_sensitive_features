''' Modules used in Fair Supervised PCA analysis'''

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import rbf_kernel


def func_read_data(data_imp, n_samp):
    ''' This function reads the considered dataset '''
    
    if data_imp == 'adult':
        # Adult dataset
        # We removed education as it reflects educational-num
        # We removed fnlwgt according to paper's suggestion
        # We considered age={25–60, <25 or >60}
        # We considered workclass={private,non-private}
        # We considered marital-status={married,never-married,other}
        # We considered occupation={office,heavy-work,service,other}
        # We considered race={white,non-white}
        # We considered native-country={US,non-US}

        dataset = pd.read_csv('data_adult.csv', na_values='?').dropna()
        X = dataset.iloc[:, 0:-1]
        X = X.drop('fnlwgt',axis=1)
        X = X.drop('education',axis=1)
        age1, age2, age3 = X.age<25, X.age.between(25, 60), X.age>60
        X['age'] = np.select([age1, age2, age3], ['<25', '25-60', '>60'], default=None)
        X['workclass'] = np.where(X['workclass'] != 'Private', 'Non-private', X['workclass'])
        X['marital-status'] = np.where(X['marital-status'] == 'Married-civ-spouse', 'married', X['marital-status'])
        X['marital-status'] = np.where(X['marital-status'] == 'Married-spouse-absent', 'married', X['marital-status'])
        X['marital-status'] = np.where(X['marital-status'] == 'Married-AF-spouse', 'married', X['marital-status'])
        X['marital-status'] = np.where(X['marital-status'] == 'Never-married', 'never-married', X['marital-status'])
        X['marital-status'] = np.where(X['marital-status'] == 'Divorced', 'other', X['marital-status'])
        X['marital-status'] = np.where(X['marital-status'] == 'Separated', 'other', X['marital-status'])
        X['marital-status'] = np.where(X['marital-status'] == 'Widowed', 'other', X['marital-status'])
        X['occupation'] = np.where(X['occupation'] == 'Adm-clerical', 'office', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Exec-managerial', 'office', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Craft-repair', 'heavy-work', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Farming-fishing', 'heavy-work', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Handlers-cleaners', 'heavy-work', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Machine-op-inspct', 'heavy-work', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Transport-moving', 'heavy-work', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Other-service', 'service', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Priv-house-serv', 'service', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Protective-serv', 'service', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Tech-support', 'service', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Prof-specialty', 'other', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Armed-Forces', 'other', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Sales', 'other', X['occupation'])
        #X['race'] = np.where(X['race'] != 'White', 'Non-white', X['race'])
        X['native-country'] = np.where(X['native-country'] != 'United-States', 'Non-United-States', X['native-country'])

        y =  dataset.loc[:,'income']=='>50K'
        y = 2*y.astype(int)-1
            
    if data_imp == 'compas':
        # Compas dataset
        
        dataset = pd.read_excel('data_compas.xlsx')
        X = dataset.iloc[:, 0:8]
        X['race'] = np.where(X['race'] == 'Hispanic', 'Other', X['race'])
        X['race'] = np.where(X['race'] == 'Asian', 'Other', X['race'])
        X['race'] = np.where(X['race'] == 'Native American', 'Other', X['race'])
        y =  dataset.loc[:,'score_risk']
        y = 2*y-1
        
    if data_imp == 'lsac_new':
        # LSAC dataset (new)
          
        dataset = pd.read_csv('data_lsac_new.csv')
        dataset[['fulltime','male','race']]=dataset[['fulltime','male','race']].astype(str)
        X = dataset.iloc[:, 0:-1]
        y =  dataset.loc[:,'pass_bar']
        y = 2*y-1
        
    if data_imp == 'tcred':
        # Taiwanese credit default dataset
        # We removed rows whose EDUCATION are 0, 5 or 6 (unknown)
        # We removed rows whose MARRIAGE is 0 (unknown)
        # We considered age={36–60, <36 or >60}

        dataset = pd.read_csv('data_tcred.csv')
        X = dataset.iloc[:, 1:]
        X = X[X.EDUCATION != 0]
        X = X[X.EDUCATION != 4]
        X = X[X.EDUCATION != 5]
        X = X[X.EDUCATION != 6]
        X = X[X.MARRIAGE != 0]
        X = X[X.MARRIAGE != 3]
        sex1, sex2 = X.SEX==1, X.SEX==2
        X['SEX'] = np.select([sex1, sex2], ['male', 'female'], default=None)
        educ1, educ2, educ3, educ4 = X.EDUCATION==1, X.EDUCATION==2, X.EDUCATION==3, X.EDUCATION==4
        X['EDUCATION'] = np.select([educ1, educ2, educ3, educ4], ['graduate school', 'university', 'high school', 'others'], default=None)
        marr1, marr2, marr3 = X.MARRIAGE==1, X.MARRIAGE==2, X.MARRIAGE==3
        X['MARRIAGE'] = np.select([marr1, marr2, marr3], ['married', 'single', 'others'], default=None)
        #age1, age2, age3 = X.AGE<35, X.AGE.between(35, 60), X.AGE>60
        #X['AGE'] = np.select([age1, age2, age3], ['<35', '35-60', '>60'], default=None)
        age1, age2 = X.AGE<35, X.AGE>=35
        X['AGE'] = np.select([age1, age2], ['<35', '>=35'], default=None)
        y =  X.loc[:,'default'].astype(float)
        X = X.drop('default',axis=1)
        y = 2*y-1
       

    if n_samp > 0:
        samp = random.sample(range(X.shape[0]), n_samp)
        X_hsic = X.iloc[samp,:]
        y_hsic = y.iloc[samp]
    else:
        X_hsic = X
        y_hsic = y
        
    
    return X,y,X_hsic,y_hsic


def func_kernel(Y, param_hsic):
    ' This function calculates the centered linear kernel of Y - Delta kernel is also available'
    n = len(Y)
    Y = np.reshape(np.array(Y), (n,1))
    H = np.eye(n) - (1/n)*np.ones((n,n))

    if(param_hsic == 'linear'):
      Hsic = Y @ Y.T #+ np.eye(n)  

    elif(param_hsic == 'delta'):
      A = np.array(Y @ Y.T)
      rows, cols = np.where(A == -1)
      A[rows, cols] = 0  
      Hsic = A + np.eye(n)
      
    else:
      Hsic = Y @ Y.T + np.eye(n)

    return H @ Hsic @ H

def func_kernel_rbf(Y):
    ' This function calculates the centered RBF kernel of Y'
    n = len(Y)
    Y = np.reshape(np.array(Y), (n,1))
    H = np.eye(n) - (1/n)*np.ones((n,n))

    Hsic = rbf_kernel(Y)
    
    return H @ Hsic @ H


def func_metrics(tp_g, fp_g, tn_g, fn_g, tp_gout, fp_gout, tn_gout, fn_gout):
    ''' This function calculates the adopted fairness metrics '''
    
    tpr_g, fpr_g = tp_g/(tp_g+fn_g), fp_g/(fp_g+tn_g)
    tpr_g[np.isnan(tpr_g)], fpr_g[np.isnan(fpr_g)] = 0, 0
    tpr_g_mean, tpr_g_std, fpr_g_mean, fpr_g_std = np.mean(tpr_g, axis=0), np.std(tpr_g, axis=0), np.mean(fpr_g, axis=0), np.std(fpr_g, axis=0)

    tpr_gout, fpr_gout = tp_gout/(tp_gout+fn_gout), fp_gout/(fp_gout+tn_gout)
    tpr_gout[np.isnan(tpr_gout)], fpr_gout[np.isnan(fpr_gout)] = 0, 0
    tpr_gout_mean, tpr_gout_std, fpr_gout_mean, fpr_gout_std = np.mean(tpr_gout, axis=0), np.std(tpr_gout, axis=0), np.mean(fpr_gout, axis=0), np.std(fpr_gout, axis=0)

    accur_g = (tp_g + tn_g)/(tp_g + tn_g + fp_g + fn_g)
    accur_g[np.isnan(accur_g)] = 0
    accur_g_mean, accur_g_std = np.mean(accur_g, axis=0), np.std(accur_g, axis=0)

    accur_gout = (tp_gout + tn_gout)/(tp_gout + tn_gout + fp_gout + fn_gout)
    accur_gout[np.isnan(accur_gout)] = 0
    accur_gout_mean, accur_gout_std = np.mean(accur_gout, axis=0), np.std(accur_gout, axis=0)

    eq_op = np.abs(tpr_g - tpr_gout)
    eq_op_mean, eq_op_std = np.mean(eq_op, axis=0), np.std(eq_op, axis=0)

    pr_pa = np.abs(fpr_g - fpr_gout)
    pr_pa_mean, pr_pa_std = np.mean(pr_pa, axis=0), np.std(pr_pa, axis=0)

    eq_od = np.abs(tpr_g - tpr_gout) + np.abs(fpr_g - fpr_gout)
    eq_od_mean, eq_od_std = np.mean(eq_od, axis=0), np.std(eq_od, axis=0)

    ov_ac =  np.abs(accur_g - accur_gout)
    ov_ac_mean, ov_ac_std = np.mean(ov_ac, axis=0), np.std(ov_ac, axis=0)

    di_im = ((tp_gout + fp_gout)/(tp_gout+tn_gout+fp_gout+fn_gout))/((tp_g + fp_g)/(tp_g+tn_g+fp_g+fn_g))
    di_im_mean, di_im_std = np.mean(di_im, axis=0), np.std(di_im, axis=0)

    return tpr_g_mean, tpr_g_std, fpr_g_mean, fpr_g_std, tpr_gout_mean, tpr_gout_std, fpr_gout_mean, accur_g_mean, accur_g_std, accur_gout_mean, accur_gout_std, eq_op_mean, eq_op_std, pr_pa_mean, pr_pa_std, eq_od_mean, eq_od_std, ov_ac_mean, ov_ac_std, di_im_mean, di_im_std


def func_max_depend(depend_max,features_encoded_max):
    ''' In this function, we plot the maximum NOCCO values '''
    order_ind = np.argsort(-depend_max).astype(int)
    order = depend_max[order_ind]
    names = list()
    for ii in range(len(depend_max)):
        names.append(features_encoded_max[order_ind[ii]])
    plt.bar(names,order)
    plt.xticks(np.arange(len(depend_max)), names, rotation=90, fontsize='11')
    plt.ylabel('Dependence measure (NOCCO)', fontsize='12')
    plt.xlabel('Features and subfeatures', fontsize='12')


def func_scatter(features_categorical_selec,depend_categorical_selec,ov_ac_mean_max,eq_op_mean_max,pr_pa_mean_max,eq_od_mean_max,perf_measures):
    order = np.argsort(-depend_categorical_selec)
    order2 = np.argsort(depend_categorical_selec)
    
    index_aux = len(depend_categorical_selec)
    colors_aux = list(['black','red','blue','green'])
    colors = list()
    features_name = list()
    for i in range(len(depend_categorical_selec)):
        features_name.append(features_categorical_selec[order2[i]])
    
    for j in range(4):
        for i in range(len(depend_categorical_selec)):
            colors.append(colors_aux[j])
    
    depend_categorical_selec = depend_categorical_selec[order]
    depend_categorical_selec = np.tile(depend_categorical_selec, (4,))
    
    ov_ac_mean_max = ov_ac_mean_max[order]
    eq_op_mean_max = eq_op_mean_max[order]
    pr_pa_mean_max = pr_pa_mean_max[order]
    eq_od_mean_max = eq_od_mean_max[order]    
    performances = np.concatenate((ov_ac_mean_max,eq_op_mean_max,pr_pa_mean_max,eq_od_mean_max))
    
    s = np.tile(np.flip(np.arange(10, 20*index_aux+1, 20, dtype=int)),(4,))
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(depend_categorical_selec, performances, c=colors, s=s)
    
    handles1 = scatter.legend_elements()[0]
    
    custom = [plt.Line2D([], [], marker='.', color='black', linestyle='None',markersize=12.0),
              plt.Line2D([], [], marker='.', color='red', linestyle='None',markersize=12.0),
              plt.Line2D([], [], marker='.', color='blue', linestyle='None',markersize=12.0),
              plt.Line2D([], [], marker='.', color='green', linestyle='None',markersize=12.0)]
    
    legend1 = ax.legend(handles = custom, labels = perf_measures, loc="upper right", title="Fairness measure", bbox_to_anchor=(1.52, 1.02))
    ax.add_artist(legend1)

    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
    labels = features_name
    legend2 = ax.legend(handles, labels, loc="lower right", title="Features and subfeatures", bbox_to_anchor=(1.58, 0.10))
    
    for i in range(4):
        xi = depend_categorical_selec[index_aux*i:index_aux*i+index_aux]
        yi = performances[index_aux*i:index_aux*i+index_aux]
        
        z = np.polyfit(xi, yi, 1)
        p = np.poly1d(z)
        ax.plot(xi,p(xi),"--",color=colors_aux[i], linewidth=0.8)
        

    plt.ylabel('Fairness measure', fontsize='12')
    plt.xlabel('Dependence measure (NOCCO)', fontsize='12')