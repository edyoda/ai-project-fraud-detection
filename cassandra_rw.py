import uuid
from cassandra.cqlengine import connection
from datetime import datetime
from cassandra.cqlengine.management import sync_table
import csv
from TxInfo import TxInfoModel
import pandas as pd

class CassandraReadWriteDb:
    
    def __init__(self, ip_addrs, keyspace):
        connection.setup( ip_addrs, keyspace, protocol_version=3)
    
    def sync_class_table(self, typeOfClass):
        self.typeOfClass = typeOfClass
        sync_table(typeOfClass)

    #Write CSV to cassandra
    def write_file_table(self, credit_logs):
        with open(credit_logs) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for row in csv_reader:
                self.typeOfClass.create(**dict(row))

    #Read Cassandra data to pandas
    def get_pandas_from_cassandra(self):
        tx_info = pd.DataFrame()

        for q in TxInfoModel.objects():
            d = pd.DataFrame.from_records([q.values()])
            tx_info = tx_info.append(d)

        tx_info.columns = q.keys()
        return tx_info


if __name__ == '__main__':

    cwd = CassandraReadWriteDb(ip_addrs=['172.17.0.2'], keyspace="emp")
    cwd.sync_class_table(TxInfoModel)
    #cwd.write_file_table('credit.csv')
    print(cwd.get_pandas_from_cassandra())


