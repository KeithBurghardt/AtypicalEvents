import sys
sys.path.insert(0,'/usr/local/lib/python3.7/site-packages')
import pandas as pd
import numpy as np
import scipy
import bottleneck as bn
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.feature_selection import f_regression
import sklearn
from scipy.stats import pearsonr, spearmanr
import pickle


class  mrmrFeatureSelector():
    def __init__(self, n_features = 10, verbose = 0, estimator = 'pearson'):
        self.n_features = n_features
        self.verbose = verbose
        self.estimator = estimator
        
    def fit(self, X, y):
        return self._fit(X, y)
    
    def transform(self, X):
        return self._transform(X)
    
    def fit_transform(self, X, y):
        self._fit(X, y)
        return self._transform(X)
    
    def _fit(self, X, y):
        
        self.X = X
        n, p = X.shape
        self.y = y.reshape((n,1))
        selected = []
        
        #List of all features indicies
        Feat = list(range(p))
        
        feature_score_matrix = np.zeros((self.n_features, p))
        feature_score_matrix[:] = np.nan
        selected_scores = []
        
        # ---------------------------------------------------------------------
        # FIND FIRST FEATURE
        # ---------------------------------------------------------------------
        
        #Calculate f-value between features and target variable
        (F_scores,pvalues) = f_regression(X,y)
        selected, Feat = self._add_remove(selected, Feat, bn.nanargmax(F_scores))
        selected_scores.append(bn.nanmax(F_scores))
        
        # ---------------------------------------------------------------------
        # FIND SUBSEQUENT FEATURES
        # ---------------------------------------------------------------------
        n_features = self.n_features
        while len(selected) < n_features:
            s=len(selected) - 1
            feature_score_matrix[s, Feat] = self.calculate_correlation(X, Feat, X.iloc[:,selected[-1]])
        
            # make decision based on the chosen FS algorithm
            fmm = feature_score_matrix[:len(selected), Feat]
            # if self.method == 'MRMR':
            if bn.allnan(bn.nanmean(fmm, axis = 0)):
                    break
            #MIQ or MIS "{-}"
            MRMR = F_scores[Feat] / bn.nanmean(fmm, axis=0)

            fselected = Feat[bn.nanargmax(MRMR)]
           
            selected_scores.append(bn.nanmax(bn.nanmin(fmm, axis=0)))
            selected, Feat = self._add_remove(selected, Feat, fselected)
        # ---------------------------------------------------------------------
        # SAVE RESULTS
        # ---------------------------------------------------------------------

        self.n_features_ = len(selected)
        self.support_ = np.zeros(p, dtype=np.bool)
        self.support_[selected] = 1
        self.ranking_ = selected
        self.scores_ = selected_scores

        return self
        
    def _transform(self, X):
        # sanity check
        try:
            self.ranking_
        except AttributeError:
            raise ValueError('You need to call the fit(X, y) method first.')
        X = X.iloc[:,self.support_]
        return X
    
    def _add_remove(self, S, F, i):
        """
        Helper function: removes ith element from F and adds it to S.
        """

        S.append(i)
        F.remove(i)
        return S, F

    def _print_results(self, S, MIs):
        out = ''
        out += ('Selected feature #' + str(len(S)) + ' / ' +
                    str(self.n_features) + ' : ' + str(S[-1]))

        if self.verbose > 1:
            out += ', ' + self.method + ' : ' + str(MIs[-1])
        print (out)
        
    def calculate_correlation(self, X, F, s):
        #should return positive numbers always
        if self.estimator == 'pearson':
            res = [abs(scipy.stats.pearsonr(s, X.iloc[:,f])[0]) for f in F]
            return res
            
        elif self.estimator == 'spearman':
            res = [abs(scipy.stats.spearmanr(s, X.iloc[:,f])[0]) for f in F]
            return res
        else:
            print("Estimator is not supported, please select pearson or spearman :)")
        


"""
Feature Selection Function

Input:
path_data: path of feature matrix file
dm_file: filename of feature matrix
path_GT: path of ground truth file
cc_file: filename of ground truth file
label: name of target variable 
features: modalities are used
i_cv: inner cross validation folds
o_cv: outer cross validation folds

Output:
selected_features: List of selected features

"""
def featureSelection(path_data,dm_file,path_GT,cc_file,label,features,i_cv,o_cv,quality_file):
    ##load the feature matrix
    df_feature=pd.read_csv(path_data+dm_file)
 
    ##load the ground truth
    df_y=pd.read_csv(path_GT+cc_file)
    if quality:
        df_y2 = pd.read_csv(quality_file)
        df_feature['day'] =pd.Series([day.split(' ')[0] for day in df_feature['Timestamp'].values])
        feature_uidday = [str(uidday) for uidday in df_feature.loc[:,['uid','day']].values]
        y_uidday = [str(uidday) for uidday in df_y2.loc[:,['uid','day']].values]
        quality_row = [True if uidday in y_uidday else False for uidday in feature_uidday]
        print(sum([1 if q else 0 for q in quality_row]))
        df_feature = df_feature.loc[quality_row,]
        df_y = df_y.loc[quality_row,]
        
    print(label in df_y.columns)
    print(df_y.columns)
    # all non-extreme atypical events are ignored
    if label == 'CoarsenedAtypicalEventCategory':
        for i in range(2,10):
            df_y[label]=df_y[label].replace(i,0)

    # feature: replace +/- inf with nan
    df_feature=df_feature.replace([np.inf, -np.inf], np.nan)
    # remove columns with NA
    #df_feature = df_feature.dropna(axis=1)
    # feature: replace all nan with mean of column
    df_feature=df_feature.fillna(df_feature.mean())
    
    # filter by column
    df_f=df_feature.filter(regex=features[0])
    for f in features[1:]:
        #if(f=='OMSignal'):
        #    om_all=df_feature.filter(regex='OMSignal').keys().tolist()
        #    om_hrv=df_feature.filter(regex='OMSignal-HRV').keys().tolist()
        #    om_features=list(set(om_all)-set(om_hrv))
        #    df_f=df_feature[om_features]
        #    df_f=df_f.join(df_feature[om_features])
        #else:
        df_f=df_f.join(df_feature.filter(regex=f))
        
    df_f['file_id']=df_feature['file_id']
    selected_features=df_f.keys().tolist()
    # removing GPS info
    selected_features = [c for c in selected_features if 'latitude' not in c and 'longitude' not in c]
    df_f = df_f.loc[:,selected_features]
    df_f=df_f.dropna(how='all',axis=1)
    df_f=df_f.dropna()
    
    #delete the nan GT
    ids=[float(round(i,1)) for i in df_y['file_id'].tolist() if not df_y[df_y.file_id==i][label].isnull().all()]
    df_f=df_f[df_f['file_id'].isin(ids)]
    
   
    #matching the corresponding GT
    yid=df_f['file_id'].tolist()
    df_y=df_y[df_y['file_id'].isin(yid)]
    users=df_y[df_y['file_id'].isin(yid)]['uid'].unique().tolist()
    users=np.array(users)
    
    #nested CV
    inner_cv=KFold(n_splits=i_cv, shuffle=True)
    outer_cv=KFold(n_splits=o_cv, shuffle=True)

    #regr=Ridge(alpha=1.0)
    #outer--cv
    rlist=[]
    num=0
    feature_freq={}
    for train_users, test_users in outer_cv.split(users):
        train_index=df_y[df_y['uid'].isin(users[train_users])]['file_id'].tolist()
        test_index=df_y[df_y['uid'].isin(users[test_users])]['file_id'].tolist()
        X_train, X_test = df_f[df_f['file_id'].isin(train_index)], df_f[df_f['file_id'].isin(test_index)]
        y_train, y_test = df_y[df_y['file_id'].isin(train_index)][label].dropna(), df_y[df_y['file_id'].isin(test_index)][label].dropna()
        y_train=np.array(y_train)
        y_test=np.array(y_test)
        X_train=X_train.drop(['file_id'],axis=1)
        X_test=X_test.drop(['file_id'],axis=1)
        feats=X_train.keys().tolist()
        X_train=normalize(X_train,norm='max',axis=0)
        X_test=normalize(X_test,norm='max',axis=0)
        
        feat_selector = mrmrFeatureSelector(n_features = 30)
        feat_fit = feat_selector.fit(pd.DataFrame(X_train), y_train)
        sf=feat_fit.ranking_
        final_features=[feats[i] for i in sf]
        X_trans=X_test[:,sf]

        for f in final_features:
            if(f in feature_freq.keys()):
                    feature_freq[f]+=1
            else:
                feature_freq.setdefault(f,1)
    
    selected_features=[k for k,v in feature_freq.items() if v>5]
    return selected_features



"""
Feature Selection Function

Input:
path_data: path of feature matrix file
dm_file: filename of feature matrix
path_GT: path of ground truth file
cc_file: filename of ground truth file
label: name of target variable
features: modalities are used
i_cv: inner cross validation folds
o_cv: outer cross validation folds

Output:
selected_features: List of selected features

"""
def featureSelectionMultiClass(path_data,dm_file,path_GT,cc_file,label,features,i_cv,o_cv,dropZeroCat,topk):
    ##load the feature matrix
    df_feature=pd.read_csv(path_data+dm_file)
    ##load the ground truth
    df_y=pd.read_csv(path_GT+cc_file)

    # feature: replace +/- inf with nan
    df_feature=df_feature.replace([np.inf, -np.inf], np.nan)
    # remove columns with NA
    #df_feature = df_feature.dropna(axis=1)
    # feature: replace all nan with mean of column
    df_feature=df_feature.fillna(df_feature.mean())

    # filter by column
    df_f=df_feature.filter(regex=features[0])
    for f in features[1:]:
        #if(f=='OMSignal'):
        #    om_all=df_feature.filter(regex='OMSignal').keys().tolist()
        #    om_hrv=df_feature.filter(regex='OMSignal-HRV').keys().tolist()
        #    om_features=list(set(om_all)-set(om_hrv))
        #    df_f=df_feature[om_features]
        #    df_f=df_f.join(df_feature[om_features])
        #else:
        df_f=df_f.join(df_feature.filter(regex=f))

    df_f['file_id']=df_feature['file_id']
    selected_features=df_f.keys().tolist()
    df_f=df_f.dropna(how='all',axis=1)
    df_f=df_f.dropna()


    #delete the nan GT
    ids=[float(round(i,1)) for i in df_y['file_id'].tolist() if not df_y[df_y.file_id==i][label].isnull().all()]
    # drop the zero category
    if dropZeroCat:
        ids += [float(round(i,1)) for i in df_y['file_id'].tolist() if not df_y[df_y.file_id==i][label].values==0]

    df_f=df_f[df_f['file_id'].isin(ids)]



    #matching the corresponding GT
    yid=df_f['file_id'].tolist()
    df_y=df_y[df_y['file_id'].isin(yid)]
    users=df_y[df_y['file_id'].isin(yid)]['uid'].unique().tolist()
    users=np.array(users)
    X=df_f.drop(['file_id'],axis=1)
    y = df_y[label].values.flatten()
    feats=X.keys().tolist()
    
    feature_values = sklearn.feature_selection.mutual_info_classif(X.values,y)
    # find the largest features
    sorted_feature_values = sorted(feature_values,reverse=True)[:topk]
    selected_features=[f for f,v in zip(feats,feature_values) if v in sorted_feature_values] 
    print(selected_features)
    return selected_features


# remove the 0 category
dropZeroCat=True
time='day'#'6h'
quality=True



##features: modalities
#features=['FitBit','Phone-Event','App-Analytics','Jelly','OMSignal']#owl in one?
#features=['Jelly','Routine-Change','App-Analytics']
features=['FitBit']#['RealizD', 'Routine-Change', 'Phone-Event', 'Motif', 'Jelly', 'FitBit', 'Minew',  'OMSignal']#,'Embeddings']
for dataset in ['Hospital','Aerospace']:
    path_data='../Features+MGT/'+dataset+'/'
    path_GT='../Features+MGT/'+dataset+'/'
    quality_file = '../Features+MGT/'+dataset+'/gt_append.csv'

    feature_path=dataset+'/Selected Features/'


    #file for the features
    # augmented features via result = pd.merge(df, df2, how='outer', on=['uid', 'Timestamp'])
    dm_file="MGT_"+time+"_dm.csv"
    #file for ground truth
    cc_file="MGT_"+time+"_gt.csv"

    for igtb in [False]:#[True,False]:
        for demog in [False]:#true
            if dataset != 'TILES':
                features = ['FitBit','Embeddings']
            if igtb:
                features +=['IGTB']
            if demog:
                features +=['Demographic','Pre-Assessment-Survey']
            ##target variable
            #targets = [c for c in pd.read_csv(path_GT+cc_file).columns if 'mgt' in c and 'location' not in c and 'work' not in c and 'interact' not in c and 'text' not in c][::-1]
            targets = ['event_mgt','GoodCoarsenedAtypicalEventCategory','BadCoarsenedAtypicalEventCategory']#['PMCoarsenedAtypicalEventCategory']#['CoarsenedAtypicalEventCategory']
            if dataset != 'TILES':
                target = ['event_mgt']
            for label in targets:
                print(label)
            

                i_cv=10
                o_cv=10
                #if 'categories' in label.lower() or 'category' in label.lower():
                #    print('Selecting Features')
                #    for topk in [20,40]:#[10,20,40]:
                #        cc_file="MGT_day_gt_appended.csv"
                #        selected_features=featureSelectionMultiClass(path_data,dm_file,path_GT,cc_file,label,features,i_cv,o_cv,dropZeroCat,topk)
                #        write_file = feature_path+label+'_igtb='+str(igtb)+'_demog='+str(demog)+'_drop0s='+str(dropZeroCat)+'_topk='+str(topk)+'_time='+time+'.txt'
                #else:
                if dataset=='TILES':#"CoarsenedAtypicalEventCategory" in label:
                    cc_file="MGT_"+time+"_gt_appended.csv"
                selected_features=featureSelection(path_data,dm_file,path_GT,cc_file,label,features,i_cv,o_cv,quality_file)
                write_file = feature_path+label+'_igtb='+str(igtb)+'_demog='+str(demog)+'_time='+time+'.txt'
                ##save selected features into file
                with open(write_file, 'w') as fp:
                    for i in selected_features:
                        fp.write(i+'\n')

