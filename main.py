from cassandra_rw import CassandraReadWriteDb
from TxInfo import TxInfoModel
from ml_model import BuildMlPipeline

if __name__ == '__main__':

    cass_rw = CassandraReadWriteDb(ip_addrs=['172.17.0.2'], keyspace="emp")

    #Load data in cassandra from csv files
    cass_rw.sync_class_table(TxInfoModel)
    cass_rw.write_file_table('creditcard.csv')

    #Load cassandra data into pandas
    credit_data = cass_rw.get_pandas_from_cassandra()

    #Create models
    ml_pipeline = BuildMlPipeline()
    ml_pipeline.set_estimators('sgdclassifier','randomForestClassifier')
    ml_pipeline.set_scalers('standardscaler')
    ml_pipeline.set_samplers('smoteenn')
    ml_pipeline.create_pipelines()

    #Hyperparameter Configuration
    params_dict = {}
    params_dict['smote'] = {'smote__k_neighbors':[5,10,15]}
    params_dict['smoteenn'] = {'smoteenn__sampling_strategy':['auto','all','not majority']}
    params_dict['randomforestclassifier'] = {'randomforestclassifier__n_estimators':[8,12]}
    params_dict['svc'] = {'svc__kernel':['linear','rbf','poly'],'svc__C':[.1,1,10]}
    ml_pipeline.set_hyperparameters(params_dict)

    credit_data = credit_data.sample(10000)
    
    X = credit_data.drop(['Time','C'],axis=1)
    y = credit_data.C
    trainX, testX, trainY, testY = train_test_split(X,y)

    #model training
    ml_pipeline.fit(trainX,trainY)

    #Calculating model performance
    ml_pipeline.score(testX,testY)
