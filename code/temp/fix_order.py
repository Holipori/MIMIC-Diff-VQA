import pandas as pd
import json
import sys
sys.path.append('../')
from question_gen import diff_ques, create_question_set
from tqdm import tqdm

global question_set
question_set = create_question_set()



file_path = '../../mimic_pair_questions.csv'
df = pd.read_csv(file_path)

d_all = pd.read_csv('../../mimic_all.csv')
with open('../../all_diseases.json', 'r') as f:
    diseases_json = json.load(f)
diseases_df = pd.DataFrame(diseases_json)

for i in tqdm(range(len(df))):
    question_type = df.loc[i]['question_type']
    if question_type == 'difference':
        study_id = df.loc[i]['study_id']
        ref_id = df.loc[i]['ref_id']
        order_main = d_all[d_all['study_id']==int(study_id)].study_order.values[0]
        order_ref = d_all[d_all['study_id']==int(ref_id)].study_order.values[0]
        # ori_answer = df.loc[i]['answer']
        if order_ref > order_main:
            # difference
            ref_record = dict(diseases_df[diseases_df['study_id'] == str(ref_id)])
            # locate study_id in diseases_df, get the idx
            idx = diseases_df[diseases_df['study_id'] == str(study_id)].index[0]
            d_json = diseases_json[idx]

            # original answer
            # qa_pair = diff_ques(d_json, ref_record, question_set)
            # answer = qa_pair[1]

            # new answer
            idx_ref = diseases_df[diseases_df['study_id'] == str(ref_id)].index[0]
            d_json_ref = diseases_json[idx_ref]
            main_record = dict(diseases_df[diseases_df['study_id'] == str(study_id)])
            qa_pair = diff_ques(d_json_ref, main_record, question_set)
            new_answer = qa_pair[1]

            new_study_id = ref_id
            new_ref_id = study_id

            df.loc[i, 'study_id'] = new_study_id
            df.loc[i, 'ref_id'] = new_ref_id
            df.loc[i, 'answer'] = new_answer

df.to_csv('mimic_pair_questions_temp.csv', index=False)