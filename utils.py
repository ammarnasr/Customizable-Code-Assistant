import json
import joblib
import os


def get_app_properties():
    """
    Get the app properties
    """
    with open("app_properties.json", "r") as f:
        app_properties = json.load(f)
    return app_properties

def dump_data(data, file_path, driver = 'joblib'):
    """
    Dump data to file. Default driver is joblib
    Steps:
        1. Check if the file extension matches the driver
        2. Check if the directory exists, if not create it
        3. Dump the data to the file
    """
    if driver == 'joblib':
        if not file_path.endswith('.joblib'):
            file_path = file_path + '.joblib'
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        joblib.dump(data, file_path)
    return file_path