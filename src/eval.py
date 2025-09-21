import json
import sklearn.metrics as metrics
import numpy as np
import argparse

possible_labels = ['O','B-Species', 'S-Species', 'S-Biological_Molecule', 'B-Chemical_Compound', 'B-Biological_Molecule', 'I-Species', 'I-Biological_Molecule', 'E-Species', 'E-Chemical_Compound', 'E-Biological_Molecule', 'I-Chemical_Compound', 'S-Chemical_Compound']

def get_data(file_path):
    with open(file_path,"r")as fread:
        data = fread.readlines()

    tags = []
    for i,d in enumerate(data):
        d = d.replace("\n",'')
        if d == '':
            continue
        
        tag = d.split("\t")[-1]
        assert tag in possible_labels, f"Non-possible tag found : {tag} at line {i}"
        tags.append(tag)
    
    return tags

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', required=True)
    parser.add_argument('--gold_file', required=True)

    args = parser.parse_args()

    pred_data = get_data(args.pred_file)
    gold_data = get_data(args.gold_file)

    possible_labels.remove('O')
    f1_micro = metrics.f1_score(gold_data, pred_data, average="micro", labels=possible_labels)
    f1_macro = metrics.f1_score(gold_data, pred_data, average="macro", labels=possible_labels)

    print(f"f1_macro : {round(f1_macro,5)}")
    print(f"f1_micro : {round(f1_micro,5)}")
