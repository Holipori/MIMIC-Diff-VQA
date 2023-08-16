import pandas as pd
from tqdm import tqdm
import argparse
import os
def find_dicom(id, d2):
    # find dicom_id from d2 that has the same study_id and ViewCodeSequence_CodeMeaning is in ['postero-anterior', 'antero-posterior']
    dicoms = d2[d2['study_id'] == id]
    dicom = dicoms[dicoms['ViewCodeSequence_CodeMeaning'].isin(['postero-anterior', 'antero-posterior'])]
    if len(dicom) == 0:
        dicom = dicoms
    dicom_id = dicom['dicom_id'].values[0]
    view = dicom['ViewCodeSequence_CodeMeaning'].values[0]
    # dicom_id = dicom_id.values[0]['dicom_id']
    return dicom_id, view

def find_split(dicom_id, d3):
    # find 'split' from d3 that has the same dicom_id
    split = d3[d3['dicom_id'] == dicom_id]['split'].values[0]
    return split

def get_uni_csv(mimic_path): # generate file

    finding_path = os.path.join(mimic_path, 'mimic-cxr-2.0.0-chexpert.csv.gz')
    dicom_path = os.path.join(mimic_path, 'mimic-cxr-2.0.0-metadata.csv.gz')
    split_path = os.path.join(mimic_path, 'mimic-cxr-2.0.0-split.csv.gz')

    d = pd.read_csv(finding_path)
    d2 = pd.read_csv(dicom_path)
    d3 = pd.read_csv(split_path)

    d['dicom_id'] = pd.NaT
    d['view'] = pd.NaT
    d['split'] = pd.NaT
    d['study_date'] = pd.NaT


    for i in tqdm(range(len(d))):
        id = d.iloc[i]['study_id']
        dicom_id,view = find_dicom(id,d2)
        # assign dicom_id and view and split
        d.at[i,'dicom_id'] = dicom_id
        d.at[i,'view'] = view
        d.at[i,'split'] = find_split(dicom_id,d3)
        d.at[i,'study_date'] = d2[d2['dicom_id'] == dicom_id]['StudyDate'].values[0]

    d['study_date'] = pd.to_datetime(d['study_date'])
    d['study_order'] = d.groupby('subject_id')['study_date'].rank(method='min')
    d.to_csv('../mimic_all.csv', index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--mimic_path", type=str, default=None, required=True, help="path to mimic-cxr-jpg dataset")
    args = parser.parse_args()
    get_uni_csv(args.mimic_path)

if __name__ == '__main__':
    main()
    # '/home/qiyuan/2021summer/physionet.org/files/mimic-cxr-jpg/2.0.0/