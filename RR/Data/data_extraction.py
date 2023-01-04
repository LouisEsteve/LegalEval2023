# LIBRAIRIES
import os
from pathlib import Path
import sys
import re

from datetime import datetime
import json
import pandas as pd

# DATA
CUR_DIR = Path(__file__).parent
print(CUR_DIR)
BASE_DIR = os.path.dirname(Path(__file__).parent)
DATA_DIR = os.path.join(BASE_DIR, "Data")

files = [file for file in os.listdir(DATA_DIR)]

# CODE
for file in files:
    root, ext = os.path.splitext(file)
    if ext==".json":
        print(f"Fichier en cours de traitement : {file}...\n")
        f = open(BASE_DIR + os.sep + "Data" + os.sep + file)
        data = json.load(f)
        
        now = datetime.now()
        time_full = now.strftime("%Y%m%d %H%M%S ")

        df_ = []
        for i, item in enumerate(data):
            id = item["id"]
            full_text = item["data"]["text"]
            meta_info = item["meta"]["group"]
            annotations = item["annotations"]
            for i in annotations:
                result = i["result"]
                for i, item in enumerate(result):
                    text_id = item["id"]
                    text = item["value"]["text"]
                    text = re.sub("\n", "", text)
                    label = item["value"]["labels"][0]
                    df_.append([id, text_id, text, label, full_text, meta_info])

        df = pd.DataFrame(df_, columns=["doc_id", "text_id", "text", "label", "full_text", "meta"])
        print(df.tail())
        print("\n")
        filename = DATA_DIR + os.sep + root + ".csv"
        # filename = DATA_DIR + time_full + root + ".csv"
        df.to_csv(filename, sep=";", index=False)
        print(f"Fichier de sortie produit : {filename}\n")

print("Fin du traitement.")