from pymongo import MongoClient
import pathlib


def local_db():
    return MongoClient(host='localhost').twitter


def ensure_dir(dir):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

