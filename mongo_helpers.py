import pymongo
import pickle
#Obtain Mongo Client using your username and password
def get_mongo_client(mongo_db_username, mongo_db_password, cluster_name):
    mongo_connection_string = 'mongodb+srv://' + mongo_db_username + ':' + mongo_db_password + '@' + cluster_name + '-vijt9.mongodb.net/test?retryWrites=true&w=majority'
    client = pymongo.MongoClient(mongo_connection_string)
