##############################
# .json extraction into .csv #
##############################

# Customisable !!
DIR = r"C:\Users\jokkl\Documents\Masters\S3\Projet en TAL"
DATA_DIR = DIR + os.sep + "Data" # For data sub-directory if present
files = [file for file in os.listdir(DATA_DIR)]


# Libraries

import os
import re
from datetime import datetime
import json
import pandas as pd


### Functions

def get_keys(obj, prev_tier = None, keys=[]):
    if type(obj)!=type({}):
        keys.append(prev_tier)
        return keys
    else:
        tier_keys = []
        for k, v in obj.items():
            new_key = k
            tier_keys.extend(get_keys(v, new_key, []))
        return tier_keys


# Code

for file in files:
    root, ext = os.path.splitext(file)
    if ext == ".json":
        print(f"Processing {root}.{ext}...")
        f = open(DATA_DIR + os.sep + file)
        data = json.load(f)
        
        now = datetime.now()
        now_full = now.strftime("%Y%m%d %H%M%S ")
        
        # labels = [get_keys(row) for row in data]
        # labels = set([val for sublist in labels for val in sublist])

        df_ = []
        for i, item in enumerate(data):
            id = item["id"]
            annotations = item["annotations"]
            for i in annotations:
                result = i["result"]
                for i, item in enumerate(result):
                    result_id = item["id"]
                    text = item["value"]["text"]
                    text = re.sub("\n", "", text)
                    label = item["value"]["labels"][0]
                    df_.append([id, result_id, text, label])

        if "RR" in root:
            cols = ["doc_id", "result_id", "text", "RR"]
        else:
            cols = ["doc_id", "result_id", "text", "Entity"]

        df = pd.DataFrame(df_, columns=cols)
        print(df.tail())
        print("\n")
        df.to_csv(DATA_DIR + os.sep + now_full + root + ".csv", sep=";", index=False)
        