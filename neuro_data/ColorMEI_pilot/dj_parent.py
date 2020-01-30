import datajoint as datajoint
import json
import hashlib
import datajoint as dj
import importlib

class DJTableBase():

    @staticmethod
    def import_module(module_name, class_name):
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    @staticmethod
    def generate_md5_hash(tuple_dict):
        string_to_hash = ""
        for _, data in tuple_dict.items():
            string_to_hash += (str(data))
        return hashlib.md5(string_to_hash.encode()).hexdigest()

    @staticmethod
    def loads(value):
        if value != '':
            return json.loads(value)
        else:
            return dict()

    @staticmethod
    def dumps(tuple_dict, key):
        """
        Runs json.dumps on dict[key] and override the original value 
        """
        tuple_dict[key] = json.dumps(tuple_dict[key]) if tuple_dict[key] != None else ''