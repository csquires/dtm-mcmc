from pymongo import MongoClient


def local_db():
    return MongoClient(host='localhost').twitter

