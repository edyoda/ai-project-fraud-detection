import threading, logging, time
import multiprocessing
import json
from kafka import KafkaConsumer, KafkaProducer
import joblib
import pandas as pd
from cassandra_rw import CassandraReadWriteDb
from PredictTxInfo import PredictTxInfoModel

class Consumer():
    def __init__(self):
        self.model = joblib.load('model3.pipeline')
        self.cwd = CassandraReadWriteDb(ip_addrs=['172.17.0.2'], keyspace="emp")
        self.cwd.sync_class_table(PredictTxInfoModel)

        
    def run(self):
        consumer = KafkaConsumer(bootstrap_servers='localhost:9092',
                                 auto_offset_reset='earliest',
                                 consumer_timeout_ms=1000, value_deserializer=lambda m: json.loads(m.decode('ascii')))
        consumer.subscribe(['credit-card-tx'])

        while True:
            for message in consumer:
                df = pd.DataFrame.from_records([message.value])
                df = df.drop(['Time'],axis=1)
                outcome = self.model.best_estimator_.predict(df)[0]
                message.value['P'] = outcome
                self.cwd.write_json_table(message.value)

        consumer.close()
        
        
def main():
    tasks = [
        Consumer()
    ]

    for t in tasks:
        t.run()

    
        
if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s.%(msecs)s:%(name)s:%(thread)d:%(levelname)s:%(process)d:%(message)s',
        level=logging.INFO
        )
    main()
