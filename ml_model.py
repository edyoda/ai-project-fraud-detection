from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump, load

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN


class BuildMlPipeline:
    
    def __init__(self):
        pass
        
    def set_estimators(self, *args):
        estimator_db = {
            'randomForestClassifier': RandomForestClassifier(),
            'svc': SVC(),
            'sgdClassifier': SGDClassifier(),
        }
        self.estimators = list(map( lambda algo: estimator_db[algo],args))
        
    def set_scalers(self, *args):
        scaler_db = {
            'standardscaler':StandardScaler(),
            'minmaxscaler':MinMaxScaler(),
        }
        self.scalers = list(map( lambda scaler: scaler_db[scaler],args))
        
    def set_samplers(self, *args):
        sampler_db = {
            'smote':SMOTE(),
            'smoteenn':SMOTEENN(),
        }
        self.samplers = list(map( lambda sampler: sampler_db[sampler],args))
        
    def set_hyperparameters(self, params):
        self.hyperparameters = params

    
    def create_pipelines(self):
        self.model_pipelines = []
        for estimator in self.estimators:
            for sampler in self.samplers:
                for scaler in self.scalers:
                    pipeline = make_pipeline(scaler, sampler, estimator)
                    self.model_pipelines.append(pipeline)
        
    
    def fit(self, trainX, trainY):
        self.gs_pipelines = []
        for idx,pipeline in enumerate(self.model_pipelines):
            elems = list(map(lambda x:x[0] ,pipeline.steps))
            param_grid = {}
            for elem in elems:
                if elem in self.hyperparameters:
                    param_grid.update(self.hyperparameters[elem])
    
            gs = GridSearchCV(pipeline, param_grid= param_grid, n_jobs=-1, cv=5)
            gs.fit(trainX, trainY)
            #dump(gs, 'model'+idx+'.pipeline') 
            self.gs_pipelines.append(gs)
      
        
    def score(self, testX, testY):
        for idx,model in enumerate(self.gs_pipelines):
            y_pred = model.best_estimator_.predict(testX)
            print (model.best_estimator_)
            print (idx,confusion_matrix(y_true=testY,y_pred=y_pred))


import pandas as pd
from sklearn.model_selection import train_test_split
if __name__ == '__main__':

    ml_pipeline = BuildMlPipeline()
    ml_pipeline.set_estimators('randomForestClassifier')
    ml_pipeline.set_scalers('standardscaler')
    ml_pipeline.set_samplers('smote','smoteenn')
    ml_pipeline.create_pipelines()
   
    print (ml_pipeline.model_pipelines)
    
    params_dict = {}
    params_dict['smote'] = {'smote__k_neighbors':[5,10,15]}
    params_dict['smoteenn'] = {'smoteenn__sampling_strategy':['auto','all','not majority']}
    params_dict['randomforestclassifier'] = {'randomforestclassifier__n_estimators':[8,12]}
    params_dict['svc'] = {'svc__kernel':['linear','rbf','poly'],'svc__C':[.1,1,10]}

    ml_pipeline.set_hyperparameters(params_dict)

    credit_data = pd.read_csv('creditcard.csv').sample(20000)
    X = credit_data.drop(['Time','C'],axis=1)
    y = credit_data.C
    trainX, testX, trainY, testY = train_test_split(X,y)
    ml_pipeline.fit(X,y)
    ml_pipeline.score(testX,testY)
    
