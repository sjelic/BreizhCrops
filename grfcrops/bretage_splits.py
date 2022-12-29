import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

import argparse

def make_dataset_dir(path, country, name):
        if os.path.isdir(os.path.join(path, country, name)):
            filelist = [ f for f in os.listdir(os.path.join(path, country, name)) if f.endswith(".npy") ]
            for f in filelist:
                os.remove(os.path.join(os.path.join(path, country, name), f))
            os.rmdir(os.path.join(path, country, name))
            os.mkdir(os.path.join(path, country, name))
        else:
            os.mkdir(os.path.join(path, country, name))

def create_dataset(path, country, name, X, y, filter_out):

    indout = [idx for idx, cl in enumerate(y) if cl in filter_out]

    X = np.array([x for idx, x in enumerate(X) if idx not in indout])
    y = np.array([y for idx, y in enumerate(y) if idx not in indout])

    # for s in set(y_sr_cleaned):
    #     print(f"{s} : {len(list(filter(lambda x: x==s, list(y_sr_cleaned))))}")
    np.save( os.path.join(path,country,name,'y.npy'), y)
    for idx, x in enumerate(X):
        np.save(os.path.join(path,country,name,f'{idx}.npy'), x)
    return X, y

def prepare_datasets(path, country, name, Xpath, ypath, filter_out):
    X = np.load(Xpath)
    y = np.load(ypath)
    make_dataset_dir(path, country, name)
    X, y = create_dataset(path, country, name,X,y,filter_out)
    X_0102, X_0304, y_0102, y_0304 = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True, stratify=y)
    X_01, X_02, y_01, y_02 = train_test_split(X_0102, y_0102, test_size=0.5, random_state=1, shuffle=True, stratify=y_0102)
    X_03, X_04, y_03, y_04 = train_test_split(X_0304, y_0304, test_size=0.5, random_state=1, shuffle=True, stratify=y_0304)
    
    make_dataset_dir(path, country, f'{name}_01')
    _, _ = create_dataset(path, country, f'{name}_01', X_01,y_01,[])

    make_dataset_dir(path, country, f'{name}_02')
    _, _ = create_dataset(path, country, f'{name}_02', X_02,y_02,[])

    make_dataset_dir(path, country, f'{name}_03')
    _, _ = create_dataset(path, country, f'{name}_03', X_03,y_03,[])

    make_dataset_dir(path, country, f'{name}_04')
    _, _ = create_dataset(path, country, f'{name}_04', X_04,y_04,[])
    

def parse_args():
    parser = argparse.ArgumentParser(description='This function takes data directoy path `path`, `country` name and dataset `name`, path to X and y, Xpath and ypath, respectively, and list of classes to discard `filter_out`, creates subdirectory path/country/name with files idx.npy and y.npy, where idx is index of parcel/instance.')
    parser.add_argument(
        'name', type=str, default="data_serbia", help='The base name of dataset.')
    parser.add_argument(
        'country', type=str, help='The name of country')
    parser.add_argument(
        'xpath', type=str, help='The path to the X numpy array')
    parser.add_argument(
        'ypath', type=str, help='The path to the y numpy array')
    parser.add_argument(
        '-p', '--path', type=str, default='../data', help='The path of the base data directory.')
    parser.add_argument(
        '-f','--filterout', type=list, default=['not_defined', 'olives'], help='The list of classes to filter out')

    args = parser.parse_args()
    
    return args
        
if __name__=='__main__':
    args = parse_args()
    prepare_datasets(path=args.path, country=args.country, name=args.name, Xpath=args.xpath, ypath=args.ypath, filter_out=args.filterout)