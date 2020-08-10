# importing libaries ----
import sys
sys.path.insert(0,'/usr/local/lib/python3.7/site-packages')
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pylab import savefig
from datetime import datetime
#from datetime import strptime
from datetime import timedelta
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,f1_score,roc_auc_score,precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample


verbose = True
multiclass = False

def extract_data(dataset,eventtype,shuffle,quality_file,quality):
    if verbose:
        print("extracting data")
    data_subset = pd.read_csv(dataset+'/data_order.csv')#atypical_pred_tiles.csv')
    #print(data_subset.iloc[:10])
    if eventtype == 'CoarsenedAtypicalEventCategory':
        gt = pd.read_csv('../Features+MGT/'+dataset+'/MGT_day_gt_appended.csv')
        dm = pd.read_csv('../Features+MGT/'+dataset+'/MGT_day_dm.csv')
        
        dm = dm.loc[~np.isnan(gt['event_mgt'])&~np.isnan(gt['CoarsenedAtypicalEventCategory']),]
        gt = pd.read_csv('../Features+MGT/'+dataset+'/MGT_day_gt_appended.csv').loc[~np.isnan(gt['event_mgt'])&~np.isnan(gt['CoarsenedAtypicalEventCategory']),['uid','Timestamp','event_mgt','CoarsenedAtypicalEventCategory']]
 
    else:
        if eventtype == 'PMCoarsenedAtypicalEventCategory':
            gt = pd.read_csv('../Features+MGT/'+dataset+'/MGT_day_gt_appended.csv').loc[:,['uid','Timestamp',eventtype]]
            dm = pd.read_csv('../Features+MGT/'+dataset+'/MGT_day_dm.csv')
            dm = dm.loc[~np.isnan(gt[eventtype]),]
            gt = pd.read_csv('../Features+MGT/'+dataset+'/MGT_day_gt_appended.csv').loc[~np.isnan(gt[eventtype]),['uid','Timestamp',eventtype]]

        else:
            if dataset == 'TILES':
                gt = pd.read_csv('../Features+MGT/'+dataset+'/MGT_day_gt_appended.csv').loc[:,['uid','Timestamp',eventtype]]
            else:
                gt = pd.read_csv('../Features+MGT/'+dataset+'/MGT_day_gt.csv').loc[:,['uid','Timestamp',eventtype]]
            gt['Timestamp'] = [row['Timestamp'].split(' ')[0] for n,row in gt.iterrows()]
            if dataset == 'TILES':
                gt['Timestamp'] = [str(datetime.strptime(row['Timestamp'],'%m/%d/%y').date()) for n,row in gt.iterrows()]
            gt[eventtype]=gt[eventtype].fillna(0)
            dm = pd.read_csv('../Features+MGT/'+dataset+'/MGT_day_dm.csv')
            #print([dm.loc[(gt['uid']==row['uid'])&(gt['Timestamp']==row['day']),] for n,row in data_subset.iloc[:10].iterrows()])
            
            dm = pd.concat([dm.loc[(gt['uid']==row['uid'])&(gt['Timestamp']==row['day']),] for n,row in data_subset.iterrows()]).reset_index()
            gt = pd.concat([gt.loc[(gt['uid']==row['uid'])&(gt['Timestamp']==row['day']),['uid','Timestamp',eventtype]] for n,row in data_subset.iterrows()]).reset_index()

            ##dm = dm.loc[~np.isnan(gt[eventtype]),]
            ##gt = pd.read_csv('../Features+MGT/'+dataset+'/MGT_day_gt_appended.csv').loc[~np.isnan(gt[eventtype]),['uid','Timestamp',eventtype]]

        if quality:
            df_y2 = pd.read_csv(quality_file)      
            dm['day'] =pd.Series([day.split(' ')[0] for day in dm['Timestamp'].values])
            feature_uidday = [str(uidday) for uidday in dm.loc[:,['uid','day']].values]
            y_uidday = [str(uidday) for uidday in df_y2.loc[:,['uid','day']].values]
            quality_row = [True if uidday in y_uidday else False for uidday in feature_uidday]
            dm = dm.loc[quality_row,]
            gt = gt.loc[quality_row,]
        print(np.sum(gt[eventtype].values[gt[eventtype].values == 1]))
        print(len(gt[eventtype]))

    if shuffle:
        if eventtype == 'CoarsenedAtypicalEventCategory':
            gt[['event_mgt','CoarsenedAtypicalEventCategory']] = np.random.permutation(gt[['event_mgt','CoarsenedAtypicalEventCategory']].values)
        else:
            gt[[eventtype]] = np.random.permutation(gt[[eventtype]].values)

    return [gt,dm]

def select_features(construct,dataset,igtb,demog,topk,dropZeros,time):
    # try MRMRs
    #file = dataset+'/Selected Features/event_mgt.txt';
    #if igtb:
    #    file = dataset+'/Selected Features/event_mgt_igtb=True.txt';
    #if igtb & demog:
    #print(construct != "CoarsenedAtypicalEventCategory")
    if construct != "CoarsenedAtypicalEventCategory":
        #file = dataset+'/Selected Features/'+construct+'_igtb='+str(igtb)+'_demog='+str(igtb)+'.txt';
        file = dataset+'/Selected Features/'+construct+'_igtb=False_demog=False_time='+time+'.txt'
    else:
        file = dataset+'/Selected Features/'+construct+'_igtb='+str(igtb)+'_demog='+str(igtb)+'_drop0s='+str(dropZeros)+'_topk='+str(topk)+'_time='+time+'.txt';
    features = pd.read_csv(file)
    features = list(features.iloc[:,0])
    return features
    

def create_models(num_features):
    list_of_models = []
    list_of_model_names = []
    
    logit = LogisticRegression(solver='lbfgs',max_iter=500)

    if multiclass:
        print("ASSUMING MULTI CLASS")
        logit = LogisticRegression(solver='lbfgs',max_iter=500,multi_class='multinomial')
    forest = RandomForestClassifier(n_estimators=100,max_depth=10,n_jobs=-1)
    extraforest = ExtraTreesClassifier(n_estimators=100,max_depth=10,n_jobs=-1)
    svm = SVC(gamma='auto')
    adaboost = AdaBoostClassifier(n_estimators=100)
    mlp = MLPClassifier(hidden_layer_sizes=(num_features,num_features,num_features),max_iter=500)


    list_of_models.append(logit)
    list_of_model_names.append('Logit')
    list_of_models.append(forest)
    list_of_model_names.append('RF')
    list_of_models.append(extraforest)
    list_of_model_names.append('extra_RF')
    list_of_models.append(svm)
    list_of_model_names.append('SVM')
    list_of_models.append(adaboost)
    list_of_model_names.append('Adaboost')
    list_of_models.append(mlp)
    list_of_model_names.append('MLP')
    
    return [list_of_models,list_of_model_names]

def upsample_train(dm,gt,features,train_uids,eventtype):

    if construct == 'PMCoarsenedAtypicalEventCategory':
        train_y = X[construct].values.flatten()
        train_dm = X.drop(construct,axis=1).values
    else:
        train_dm = pd.concat([dm.loc[dm['uid']== uid,features] for uid in train_uids], ignore_index=True)
        train_dm = train_dm.fillna(train_dm.mean())
        train_y = pd.concat([gt.loc[dm['uid']== uid,[eventtype]] for uid in train_uids], ignore_index=True)
    
        # concatenate our training data back together
        X = pd.concat([train_dm, train_y], axis=1)
        # separate minority and majority classes
        no_event = X[X.event_mgt==0]
        atypical_event = X[X.event_mgt==1]

        # upsample minority
        event_upsampled = resample(atypical_event,
                      replace=True, # sample with replacement
                      n_samples=len(no_event) # match number in majority class
                      )
        # combine majority and upsampled minority
        upsampled = pd.concat([no_event,event_upsampled])
        train_y = upsampled.event_mgt.values.flatten()
        train_dm = upsampled.drop(eventtype, axis=1).values
    
    return [train_y,train_dm]


def downsample_train(dm,gt,features,train_uids,construct):
    train_dm = pd.concat([dm.loc[dm['uid']== uid,features] for uid in train_uids], ignore_index=True)
    train_dm = train_dm.fillna(train_dm.mean())
    train_y = pd.concat([gt.loc[dm['uid']== uid,[construct]] for uid in train_uids], ignore_index=True)
    # concatenate our training data back together
    X = pd.concat([train_dm, train_y], axis=1)
    if construct == 'CoarsenedAtypicalEventCategory':
        # remove 0 events
        X = X[X[construct] > 0]
        train_y = X[construct].values.flatten()
        train_dm = X.drop(construct,axis=1).values
    elif construct == 'PMCoarsenedAtypicalEventCategory':
        no_event = X[X[construct]==0]
        atypical_event = X[X[construct]!=0]
        event_downsampled = resample(no_event,
                      replace=False, # sample without replacement
                      n_samples=len(atypical_event) # match number in majority class
                      )
        # combine majority and upsampled minority
        downsampled = pd.concat([atypical_event,event_downsampled])
        train_y = downsampled[construct].values.flatten()
        train_dm = downsampled.drop(construct, axis=1).values

        #train_y = X[construct].values.flatten()
        #train_dm = X.drop(construct,axis=1).values
 
    else: # assume binomial events
        # separate minority and majority classes
        no_event = X[X[construct]==0]
        atypical_event = X[X[construct]==1]

        # upsample minority
        event_downsampled = resample(no_event,
                      replace=False, # sample without replacement
                      n_samples=len(atypical_event) # match number in majority class
                      )
        # combine majority and upsampled minority
        downsampled = pd.concat([atypical_event,event_downsampled])
        train_y = downsampled[construct].values.flatten()
        train_dm = downsampled.drop(construct, axis=1).values
    
    return [train_y,train_dm]

    
def cv_test(gt,dm,features,model,eventtype):
    num_folds = 4
    kfold = KFold(num_folds, shuffle=True)
    uids = gt['uid'].drop_duplicates().values
    f1_cv = []
    auc_cv = []
    precision_cv = []
    confusion = np.array([[0,0],[0,0]])
    if multiclass:
        confusion = np.array([[0,0,0],[0,0,0],[0,0,0]])
    for train_uids_ind,test_uids_ind in kfold.split(uids):
        train_uids = uids[train_uids_ind]
        test_uids = uids[test_uids_ind]
        
        #train_y,train_dm = upsample_train(dm,gt,features,train_uids)
        if multiclass:
            train_y,train_dm = downsample_train(dm,gt,features,train_uids,'PMCoarsenedAtypicalEventCategory')
            test_y = pd.concat([gt.loc[dm['uid']== uid,['PMCoarsenedAtypicalEventCategory']] for uid in test_uids], ignore_index=True).values.flatten()
        else:
            train_y,train_dm = downsample_train(dm,gt,features,train_uids,eventtype)
            test_y = pd.concat([gt.loc[dm['uid']== uid,[eventtype]] for uid in test_uids], ignore_index=True).values.flatten()
        
        test_dm = pd.concat([dm.loc[dm['uid']== uid,features] for uid in test_uids], ignore_index=True)
        test_dm = test_dm.fillna(test_dm.mean()).values

        model.fit(train_dm, train_y)  
        predict_y = model.predict(test_dm)
        if multiclass:
            f1 = f1_score(test_y,predict_y,average='weighted')
            auc_score = -1#roc_auc_score(test_y,predict_y,average='macro',multi_class='ovo')
            prec = precision_score(test_y,predict_y,average = 'weighted')
            confusion += confusion_matrix(test_y,predict_y)

        else:
            f1 = f1_score(test_y,predict_y)
            auc_score = roc_auc_score(test_y,predict_y)
            prec = precision_score(test_y,predict_y)
            confusion += confusion_matrix(test_y,predict_y)

        f1_cv.append(f1)
        auc_cv.append(auc_score)
        precision_cv.append(prec)
    
    mean_f1 = np.mean(f1_cv)
    mean_auc= np.mean(auc_cv)
    mean_precision = np.mean(precision_cv)

    std_f1 = np.std(f1_cv)
    std_auc = np.std(auc_cv)
    std_precision = np.std(precision_cv)
    print([mean_f1,std_f1,mean_auc,std_auc,mean_precision,std_precision])
    print("Precision: "+str(mean_precision)+"+/-"+str(std_precision))
    #confusion/=num_folds
        
    return [mean_auc,std_auc,mean_f1,std_f1,mean_precision,std_precision,list(confusion)]#[mean_f1,std_f1,list(confusion)]

def cv_test_atypical_cat(gt,dm,event_features,event_model,cat_features,cat_model):
    num_folds = 10
    kfold = KFold(num_folds, shuffle=True)
    uids = gt['uid'].drop_duplicates().values
    f1_cv = []
    rand_f1_cv = []
    auc_cv = []
    #confusion_event = np.array([[0,0],[0,0]])
    confusion = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    
    for train_uids_ind,test_uids_ind in kfold.split(uids):
        train_uids = uids[train_uids_ind]
        test_uids = uids[test_uids_ind]
        
        #train_y,train_dm = upsample_train(dm,gt,features,train_uids)
        train_event_y,train_event_dm = downsample_train(dm,gt,event_features,train_uids,eventtype)
        train_cat_y,train_cat_dm = downsample_train(dm,gt,cat_features,train_uids,'CoarsenedAtypicalEventCategory')
        
        event_model.fit(train_event_dm, train_event_y)  
        cat_model.fit(train_cat_dm, train_cat_y)  
        
        test_dm_event = pd.concat([dm.loc[dm['uid']== uid,event_features] for uid in test_uids], ignore_index=True)
        test_dm_cat = pd.concat([dm.loc[dm['uid']== uid,cat_features] for uid in test_uids], ignore_index=True)

        test_dm_event = test_dm_event.fillna(test_dm_event.mean()).values
        test_dm_cat = test_dm_cat.fillna(test_dm_cat.mean()).values

        test_y = pd.concat([gt.loc[dm['uid']== uid,['CoarsenedAtypicalEventCategory']] for uid in test_uids], ignore_index=True).values.flatten()
        #predict if there is an event
        predict_event = event_model.predict(test_dm_event)
        # predict category
        predict_cat = cat_model.predict(test_dm_cat)
        # set categories to 0 if we predict no event
        predict_y = [c if e > 0 else 0 for e,c in zip(predict_event,predict_cat)]
        f1 = f1_score(test_y,predict_y,labels=[0,1,2,3],average='weighted')
        f1_rand = f1_score(test_y,np.random.permutation(predict_y),labels=[0,1,2,3],average='weighted')
        f1_cv.append(f1)
        rand_f1_cv.append(f1_rand)
        #auc = roc_auc_score(test_y,predict_y,average='weighted')
        #auc_cv.append(auc)
        #confusion += confusion_matrix(test_y,predict_y)
    
    rand_f1 = np.mean(rand_f1_cv)
    mean_f1 = np.mean(f1_cv)
    std_f1 = np.std(f1_cv)

    #mean_auc = np.mean(auc_cv)
    #std_auc = np.std(auc_cv)

    #confusion/=num_folds
        
    return [rand_f1,mean_f1,std_f1,list(confusion)]

def downsample_train_rand(dm,gt,features,train_ind,construct):
    train_y = gt.iloc[train_ind].loc[:,construct]  
    train_dm = dm.iloc[train_ind].loc[:,features]
    train_dm = train_dm.fillna(train_dm.mean())
        
    #train_dm = pd.concat([dm.loc[dm['uid']== uid,features] for uid in train_uids], ignore_index=True)
    #train_dm = train_dm.fillna(train_dm.mean())
    #train_y = pd.concat([gt.loc[dm['uid']== uid,[construct]] for uid in train_uids], ignore_index=True)
    # concatenate our training data back together
    X = pd.concat([train_dm, train_y], axis=1)
    if construct == 'CoarsenedAtypicalEventCategory':
        # remove 0 events
        X = X[X[construct] > 0]
        train_y = X[construct].values.flatten()
        train_dm = X.drop(construct,axis=1).values
    elif construct == 'PMCoarsenedAtypicalEventCategory':
        no_event = X[X[construct]==0]
        atypical_event = X[X[construct]!=0]
        event_downsampled = resample(no_event,
                      replace=False, # sample without replacement
                      n_samples=len(atypical_event) # match number in majority class
                      )
        # combine majority and upsampled minority
        downsampled = pd.concat([atypical_event,event_downsampled])
        train_y = downsampled[construct].values.flatten()
        train_dm = downsampled.drop(construct, axis=1).values

        #train_y = X[construct].values.flatten()
        #train_dm = X.drop(construct,axis=1).values
 
    else: # assume binomial events
        # separate minority and majority classes
        no_event = X[X[construct]==0]
        atypical_event = X[X[construct]==1]

        # upsample minority
        event_downsampled = resample(no_event,
                      replace=False, # sample without replacement
                      n_samples=len(atypical_event) # match number in majority class
                      )
        # combine majority and upsampled minority
        downsampled = pd.concat([atypical_event,event_downsampled])
        train_y = downsampled[construct].values.flatten()
        train_dm = downsampled.drop(construct, axis=1).values
    
    return [train_y,train_dm]

def cv_test_rand(gt,dm,features,model,eventtype):
    num_folds = 4
    kfold=StratifiedKFold(n_splits=10, random_state=7, shuffle=True)

    #kfold = KFold(num_folds, shuffle=True)
    #uids = gt['uid'].drop_duplicates().values
    f1_cv = []
    auc_cv = []
    precision_cv = []
    confusion = np.array([[0,0],[0,0]])
    fold = 0
    predictions = np.array([0]*len(gt))
    for train_ind,test_ind in kfold.split(dm,gt[eventtype]):
        #print(len(predictions))
        fold+=1
        train_y,train_dm = downsample_train_rand(dm,gt,features,train_ind,eventtype)
        test_y = gt.iloc[test_ind].loc[:,[eventtype]].values.flatten()    
        test_dm = dm.iloc[test_ind].loc[:,features]
        test_dm = test_dm.fillna(test_dm.mean()).values
        model.fit(train_dm, train_y)  
        predict_y = model.predict(test_dm)
        predictions[test_ind] = predict_y[:]
        f1 = f1_score(test_y,predict_y)
        auc_score = roc_auc_score(test_y,predict_y)
        prec = precision_score(test_y,predict_y)
        confusion += confusion_matrix(test_y,predict_y)

        f1_cv.append(f1)
        auc_cv.append(auc_score)
        precision_cv.append(prec)
    
    mean_f1 = np.mean(f1_cv)
    mean_auc= np.mean(auc_cv)
    mean_precision = np.mean(precision_cv)

    std_f1 = np.std(f1_cv)
    std_auc = np.std(auc_cv)
    std_precision = np.std(precision_cv)
    print([mean_f1,std_f1,mean_auc,std_auc,mean_precision,std_precision])
    print("Precision: "+str(mean_precision)+"+/-"+str(std_precision))
    #confusion/=num_folds
    #print(len(predictions))
    return [mean_auc,std_auc,mean_f1,std_f1,mean_precision,std_precision,list(confusion),predictions]#[mean_f1,std_f1,list(confusion)]

def eval_models(gt,dm,features,eval_file,eventtype,split_uid):
    num_features = len(features)
    models,model_names = create_models(num_features)

    eval_metrics = {name:[] for name in model_names}
    max_auc=0.0
    #best_predictions = []
    for model,name in zip(models,model_names):
        print(name)
        if split_uid:
            mean_auc,std_auc,mean_f1,std_f1,mean_prec,std_prec,confusion = cv_test(gt,dm,features,model,eventtype)
            predictions = gt[eventtype]
        else:
            mean_auc,std_auc,mean_f1,std_f1,mean_prec,std_prec,confusion,predictions = cv_test_rand(gt,dm,features,model,eventtype)
        eval_metrics[name] = [mean_auc,std_auc,mean_f1,std_f1,mean_prec,std_prec,confusion]
        print([mean_auc, max_auc])
        print(mean_auc > max_auc)
        if mean_auc > max_auc:
            max_auc=mean_auc
            best_predictions = predictions[:]
            print('GT: ',len(gt),' pred: ',len(best_predictions))

    pd.DataFrame(data=eval_metrics).to_csv(eval_file,index=False)
    gt['pred']=best_predictions
    #print(gt.iloc[:10])
    if split_uid:
        gt.to_csv(eval_file[:-4]+'_agg_predictions.csv',index=False)
    return eval_metrics

def eval_atypical_cat_models(gt,dm,event_features,cat_features,eval_file):
    num_features_event = len(event_features)
    num_features_cat = len(cat_features)
    event_models,event_model_names = create_models(num_features_event)
    cat_models,cat_model_names = create_models(num_features_cat)

    eval_metrics = {}
    for model,name in zip(event_models,event_model_names):
        for cat_model,cat_name in zip(cat_models,cat_model_names):
            if cat_name != "Logit":
                print([name,cat_name])
                rand_f1,mean_f1,std_f1,confusion = cv_test_atypical_cat(gt,dm,event_features,model,cat_features,cat_model)
                eval_metrics[name+"_"+cat_name] = [rand_f1,mean_f1,std_f1,confusion]
    pd.DataFrame(data=eval_metrics).to_csv(eval_file,index=False)
    return eval_metrics


def main():
    dropZeros = True
    quality = False
    split_uid=False
    igtb=False
    demog=False
    for dataset in ['Hospital','Aerospace']:
        for time in ['day']:#,'6h']:
            for shuffle in [False]:#[True,False]:
                for topk in [10]:#[10,20,40]:
                    if dataset == 'TILES':
                        quality_file = '../Features+MGT/'+dataset+'/gt_append.csv'
                    else:
                        quality_file = '../Features+MGT/'+dataset+'/MGT_day_gt.csv'
                    eventtypes = ["event_mgt","GoodCoarsenedAtypicalEventCategory","BadCoarsenedAtypicalEventCategory"]
                    if dataset != 'TILES':
                        eventtypes = ["event_mgt"]
                            for event in eventtypes:
                                print([dataset,time,topk,igtb,demog,event])
                                eval_file = dataset+'/eval_metrics_igtb='+str(igtb)+'_demog='+str(demog)+'_event='+event+'_shuffle='+str(shuffle)+'_time='+time+'_FitBitOnly.csv'
                                if event == "CoarsenedAtypicalEventCategory":
                                    eval_file = dataset+'/eval_metrics_igtb='+str(igtb)+'_demog='+str(demog)+'_event='+event+'_shuffle='+str(shuffle)+'_topk='+str(topk)+'_time='+time+'.csv'
                                # 1. choose features (impute missing data)
                                event_features=select_features(event,dataset,igtb,demog,topk,dropZeros,time)

                                if event == "PMCoarsenedAtypicalEventCategory":
                                    eval_file = dataset+'/eval_metrics_igtb='+str(igtb)+'_demog='+str(demog)+'_event='+event+'_shuffle='+str(shuffle)+'_MRMR=True_time='+time+'.csv'
                                    # 1. choose features (impute missing data)
                                    event_features=select_features(event,dataset,igtb,demog,topk,dropZeros,time)

                                # extract data
                                gt,dm = extract_data(dataset,event,shuffle,quality_file,quality)
                                # do CV for different models
                                if verbose:
                                    print('Extracting data from '+dataset)
                                # cross-validated train-test data -> find what model performs best
                                evals = eval_models(gt,dm,event_features,eval_file,event,split_uid)

if __name__ == "__main__":
    main()
