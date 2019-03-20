#!flask/bin/python
from flask import Flask, jsonify, request
import joblib
import pandas as pd
from cassandra_rw import CassandraReadWriteDb
from PredictTxInfo import PredictTxInfoModel

app = Flask(__name__)
cwd = CassandraReadWriteDb(ip_addrs=['172.17.0.2'], keyspace="emp")
cwd.sync_class_table(PredictTxInfoModel)

@app.route('/predict/tx', methods=['POST'])
def create_task():
    tx_data = request.json
    df = pd.DataFrame.from_records([tx_data])
    df = df.drop(['Time'],axis=1)
    model = joblib.load('model3.pipeline')
    tx_data['P'] = model.best_estimator_.predict(df)[0]
    cwd.write_json_table(tx_data)
    tx_data['P'] = str(tx_data['P'])
    return jsonify(tx_data), 201

if __name__ == '__main__':
    app.run(debug=True)
