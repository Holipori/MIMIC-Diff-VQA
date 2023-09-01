# the real file for generating the question
import json
import os
import nltk
import spacy
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import random
import h5py
import sys
import pickle

import argparse



nltk.download('averaged_perceptron_tagger')

def check_any_in(list, text):
    for word in list:
        if word in text:
            return word
    return False

def get_label(caption_list, max_seq):
    output = np.zeros(max_seq)
    output[:len(caption_list)] = np.array(caption_list)
    return output

def find_report(study_id, report_path):
    path = '../mimic_all.csv'
    df_all = pd.read_csv(path)
    subject_id = df_all[df_all['study_id'] == int(study_id)]['subject_id'].values[0]
    p1 = 'p' + str(subject_id)[:2]
    p2 = 'p'+str(subject_id)
    report_name = 's' + str(int(study_id)) + '.txt'
    with open(os.path.join(report_path, p1, p2, report_name), 'r') as f:
        report = f.read()

    report.replace('\n', '').replace('FINDINGS', '\nFINDINGS').replace('IMPRESSION', '\nIMPRESSION')
    return report

def sub_find_attribute(re_searched, all_prep_words,anchor, dict_attributes, print_test):
    if re_searched is not None:
        text_pre = re_searched.group(1).strip()
        if anchor in text_pre: # todo: inner loop of detecting multiple findings (see the other todo for details)
            re_searched2 = re.search('(.*)' + anchor + '(.*)', text_pre, re.I)
            dict_attributes =  sub_find_attribute(re_searched2, all_prep_words, anchor, dict_attributes, print_test)
            text_pre = re_searched2.group(2).strip()
        if text_pre.split(' ')[-1] in all_prep_words:
            text_pre = ''
        while check_any_in(all_prep_words, text_pre):
            re_searched2 = re.search('(in|of|with) (.*)', text_pre, re.I) # todo. add 'and'
            if re_searched2 is not None:
                text_pre = re_searched2.group(2)
            else:
                break


        if check_any_in(phrases_list, text_pre):
            phrase = check_any_in(phrases_list, text_pre)
            text_list = [phrase]
            rest_list = text_pre.split(phrase)
            for text in rest_list:
                if text != '':
                    text_list += text.strip().split(' ')
        else:
            text_list = text_pre.strip().split(' ')[-5:]


        for word in text_list:
            if word == '':
                continue
            if len(d_loc[d_loc['location'] == word].values) != 0:
                if dict_attributes[anchor]['location'] == None:
                    dict_attributes[anchor]['location'] = [word]
                else:
                    if word not in dict_attributes[anchor]['location']:
                        dict_attributes[anchor]['location'].append(word)
                if print_test:
                    print('location:', word)
            elif len(d_t[d_t['type'] == word].values) != 0:
                if dict_attributes[anchor]['type'] == None:
                    dict_attributes[anchor]['type'] = [word]
                else:
                    if word not in dict_attributes[anchor]['type']:
                        dict_attributes[anchor]['type'].append(word)
                if print_test:
                    print('type:', word)
            elif len(d_lev[d_lev['level'] == word].values) != 0:
                if dict_attributes[anchor]['level'] == None:
                    dict_attributes[anchor]['level'] = [word]
                else:
                    if word not in dict_attributes[anchor]['level']:
                        dict_attributes[anchor]['level'].append(word)
                if print_test:
                    print('level:', word)
            else:
                if print_test:
                    print('pre:', word)
    return dict_attributes

def find_pre_attribute(match_words, prep, text,dict_attributes, print_test):
    re_searched = re.search('(.*) ' + match_words[0], text, re.I)
    dict_attributes = sub_find_attribute(re_searched, prep, match_words[0], dict_attributes, print_test)
    for i in range(len(match_words) - 1):
        if print_test:
            print(' ')
        word1 = match_words[i]
        word2 = match_words[i + 1]
        re_searched = re.search(word1 + '(.*)' + word2, text, re.I)
        # if 'suggest' in re_searched.group(1):
        #     print('suggest:', word1, word2)
        dict_attributes = sub_find_attribute(re_searched, prep,match_words[i+1], dict_attributes, print_test)
    return dict_attributes
def find_post_attribute(match_words, text, dict_attributes, print_test):
    keyword = ['in the', 'at the', 'seen']

    resolved_words = ['has resolved', 'have resolved']
    new_match_words = [] # in case some disease has been resolved
    for i in range(len(match_words)):
        word1 = match_words[i]
        if i +1 < len(match_words):
            word2 = match_words[i+1]
            re_searched = re.search(word1 + '(.*)' + word2, text, re.I)
        else:
            re_searched = re.search(word1 + '(.*)', text, re.I)
        text_post = re_searched.group(1).strip()

        if check_any_in(resolved_words, text_post):
            del dict_attributes[word1]
            continue
        else:
            new_match_words.append(word1)

        if check_any_in(keyword,text_post):
            phrase = check_any_in(phrases_list_post, text_post)
            if phrase:
                dict_attributes[word1]['post_location'] = phrase
                if print_test:
                    print('post_location:', phrase)
            else:
                if print_test:
                    print('post:',text_post)
    return new_match_words, dict_attributes


def create_empty_attributes(match_words):
    dict = {}
    for word in match_words:
        dict[word] = {'entity_name':word, 'location': None, 'type': None, 'level': None, 'post_location':None, 'location2':None, 'type2':None, 'level2':None, 'post_location2':None}
    return dict

def find_attribute(match_words,text, print_test):
    prep = ['and', 'in', 'of', 'with']
    dict_attributes = create_empty_attributes(match_words)
    if len(match_words) == 0:
        return dict_attributes
    match_words, dict_attributes = find_post_attribute(match_words, text, dict_attributes, print_test)
    if len(match_words)> 0:
        dict_attributes = find_pre_attribute(match_words, prep, text,dict_attributes, print_test)#main
    return dict_attributes

def reorder(match_words, indexes, text):
    # make sure the oder in match_words is corresponding to the original text
    index = text.find(match_words[-1])
    i = 0
    while i < len(indexes):
        if index < indexes[i]:
            indexes = indexes[:i] + [index] + indexes[i:]
            match_words = match_words[:i] + [match_words[-1]] + match_words[i:-1]
            break
        else:
            i += 1
    if i == len(indexes):
        indexes.append(index)
    return match_words, indexes

def get_phrases_list(d, title):
    outlist = []
    for i in range(len(d)):
        if len(d.iloc[i][title].split(' '))> 1:
            outlist.append(d.iloc[i][title])
    return outlist

def fix_order(dict_attributes):
    new_dict = {}
    # make sure the order of the attributes is consistent with d_d['official_name']
    official_names = d_d['official_name'].values
    for name in official_names:
        if name in dict_attributes:
            new_dict[name] = dict_attributes[name]

    assert len(new_dict) == len(dict_attributes)
    return new_dict


def process_core(text_core, nlp,print_test, uniform_name= False, fixed_order=False):

    if print_test:
        doc = nlp(text_core)
        print('ref_entity:', doc.ents)

    #by_match
    yes_id = set()
    match_out = []
    location = []
    mix_out= []
    indexes = []
    for i in range(len(df)):
        name = df.iloc[i]['report_name']
        # if
        if df.iloc[i]['report_name'] in text_core:
            id = df.iloc[i]['id']
            if id not in yes_id:
                yes_id.add(id)
                match_out.append(df.iloc[i]['report_name'])
                match_out, indexes = reorder(match_out, indexes, text_core)
    dict_attributes = find_attribute(match_out, text_core, print_test)
    if uniform_name:
        ori_dict = dict_attributes
        dict_attributes = {}
        for key in ori_dict:
            id = df[df['report_name'] == key]['id'].values[0]
            new_name = d_d[d_d['id'] == id]['official_name'].values[0]
            ori_dict[key]['entity_name'] = new_name
            dict_attributes[new_name] = ori_dict[key]
    if fixed_order:
        dict_attributes = fix_order(dict_attributes)
    if print_test:
        print('match_way: ', ', '.join(match_out))

    if print_test:
        missed = []
        for ent in doc.ents:
            if ent.text not in ' '.join(match_out):
                missed.append(ent.text)
        if missed!= []:
            print('missed:', ', '.join(missed))


        tokened_s = nltk.word_tokenize(text_core)
        pos = nltk.pos_tag(tokened_s)
        chanageset = {'layering', 'right', 'small', 'minimal', 'left', 'of'}
        for i in range(len(pos)):
            p = pos[i]
            if p[0] in chanageset:
                pos[i] = (p[0], 'RB')
        # print(result)
        out = ''
        outpos = []
        skipset = {'VB', 'IN', 'CC', 'VBD', 'VBG', 'VBP', 'VBZ'}
        for j in range(len(tokened_s)):
            if pos[j][1] in skipset or tokened_s[j] == ',' or tokened_s[j] == '.' or tokened_s[j] == '//':
                break
            out += tokened_s[j] + ' '
            outpos.append(pos[j])
        # if out != '':
        print('ref_nltk_way: ', out)
        # if out == 'right ' or out == 'areas ' or out == 'left ' or out == 'small ' or out == 'to ':
        #     print(s)
        #     print(pos)
    return dict_attributes

def check_matches(entities1, entities2):
    # remove the entities that are in the other
    for e1 in entities1:
        for e2 in entities2:
            if df[df['report_name'] ==e1]['id'].values[0] == df[df['report_name'] ==e2]['id'].values[0]:
                entities1.remove(e1)
    return entities1
            # if e1.text == e2.text:
            #     return True

def replace_location_words(attributes):
    for key in attributes:
        if attributes[key]['location'] is not None:
            location = ' '.join(attributes[key]['location'])
            for j in range(len(dc)):
                location = location.replace(dc.iloc[j]['from'], dc.iloc[j]['to'])
            attributes[key]['location'] = location.split(' ')

        if attributes[key]['post_location'] is not None:
            location = attributes[key]['post_location']
            for j in range(len(dc)):
                location = location.replace(dc.iloc[j]['from'], dc.iloc[j]['to'])
            attributes[key]['post_location'] = location
    return attributes

def find_better_attributes(file_attributes, sent_attributes):
    file_score = 0
    sent_score = 0
    for key in file_attributes:
        if file_attributes[key] is not None:
            file_score += 1
    for key in sent_attributes:
        if sent_attributes[key] is not None:
            sent_score += 1
    if file_score > sent_score:
        return file_attributes
    else:
        return sent_attributes

def add_new_instance(one_file_attributes, one_sent_attributes):
    for key in one_sent_attributes:
        if key not in one_file_attributes:
            one_file_attributes[key] = one_sent_attributes[key]
        else:
            sent_key_loc_word = False
            file_key_loc_word = False
            if one_sent_attributes[key]['location'] is not None:
                sent_key_loc_word = check_any_in(['left', 'right', 'bilateral', 'bibasilar'], ' '.join(one_sent_attributes[key]['location']))
            if one_file_attributes[key]['location'] is not None:
                file_key_loc_word = check_any_in(['left', 'right', 'bilateral', 'bibasilar'], ' '.join(one_file_attributes[key]['location']))
            if sent_key_loc_word and file_key_loc_word and sent_key_loc_word != file_key_loc_word:
                one_file_attributes[key]['location2'] = one_sent_attributes[key]['location']
                one_file_attributes[key]['type2'] = one_sent_attributes[key]['type']
                one_file_attributes[key]['level2'] = one_sent_attributes[key]['level']
                one_file_attributes[key]['post_location2'] = one_sent_attributes[key]['post_location'] # todo: now only support two findings of the same disease
            else:
                the_one_to_keep = find_better_attributes(one_file_attributes[key], one_sent_attributes[key])
                one_file_attributes[key] = the_one_to_keep
    return one_file_attributes

def find_general(sentences, nlp, print_test, uniform_name, fixed_order):
    one_file_positives = []
    one_file_negatives = []
    for s in sentences:
        s = s.lower()
        if print_test:
            print(' ')
            print(s)
        text_core = s.replace('     ', ' ').replace('    ', ' ').replace('   ', ' ').replace('  ', ' ')
        text_no = ''

        # definately no
        if 'no longer' in text_core or ('resolved' in text_core and 'not resolved' not in text_core) or ('disappeared' in text_core and 'not disappeared' not in text_core):
            text_no = text_core
            text_core = ''
        elif re.search('(.*) (without|no |clear of|r/o |rule out|less likely)(.*)', text_core, re.I) is not None:
            re_searched = re.search('(.*) (without|no |clear of|r/o |rule out|less likely)(.*)', text_core, re.I)
            text_core = re_searched.group(1)
            text_no = re_searched.group(3) + ' ' + text_no
            if re.search('(.*) (without|no |clear of|r/o |rule out|less likely)(.*)', text_core, re.I) is not None:
                re_searched2 = re.search('(.*) (without|no |clear of|r/o |rule out|less likely)(.*)', text_core, re.I)
                text_core = re_searched2.group(1)
                text_no2 = re_searched2.group(3)
                text_no = text_no2 + ' ' + text_no
                if re.search('(.*) (without|no |clear of|r/o |rule out|less likely)(.*)', text_core, re.I) is not None:
                    re_searched3 = re.search('(.*) (without|no |clear of|r/o |rule out|less likely)(.*)', text_core, re.I)
                    text_core = re_searched3.group(1)
                    text_no3 = re_searched3.group(3)
                    text_no = text_no3 + ' ' + text_no
                    if re.search('(.*) (without|no |clear of|r/o |rule out|less likely)(.*)', text_core, re.I) is not None:
                        re_searched4 = re.search('(.*) (without|no |clear of|r/o |rule out|less likely)(.*)', text_core, re.I)
                        text_core = re_searched4.group(1)
                        text_no4 = re_searched4.group(3)
                        text_no = text_no4 + ' ' + text_no
            if 'change in' in text_no:
                text_core = text_core + ' ' + text_no
                text_no = ''
        if text_no != '':
            no_out = process_core(text_no, nlp, print_test, uniform_name, fixed_order)
            for key in no_out:
                if one_file_negatives == []:
                    one_file_negatives = [key]
                else:
                    one_file_negatives.append(key)
            if print_test:
                print('no out:', no_out)

        one_sent_attributes = process_core(text_core, nlp,print_test, uniform_name, fixed_order)
        one_sent_attributes = replace_location_words(one_sent_attributes)
        if 'heart size is enlarged' in one_sent_attributes:
            one_sent_attributes['cardiomegaly'] = one_sent_attributes['heart size is enlarged']
            one_sent_attributes['cardiomegaly']['entity_name'] = 'cardiomegaly'
            one_sent_attributes.pop('heart size is enlarged')
        if not fixed_order:
            one_file_negatives = check_matches(one_file_negatives, one_sent_attributes)
        else:
            for key in one_file_negatives:
                if key in one_sent_attributes:
                    one_sent_attributes.pop(key)
        if one_file_positives == []:
            one_file_positives = one_sent_attributes
        else:
            one_file_positives = add_new_instance(one_file_positives, one_sent_attributes) # todo: outer loop (see the other todo for details)

        # transfrom structure
    #     out = []
    #     for k in one_file_positives:
    #         out.append(one_file_positives[k])
    # return out
    return one_file_positives, one_file_negatives

def process_postlocation(d_ploc, dc):
    for i in range(len(d_ploc)):
        text = d_ploc.iloc[i]['post_location']
        for j in range(len(dc)):
            new_text = text.replace(dc.iloc[j]['from'], dc.iloc[j]['to'])
            # if d_ploc does not have the new text, add it
            if new_text not in d_ploc['post_location'].values:
                d_ploc.loc[d_ploc.shape[0]] = [new_text, d_ploc.iloc[i]['relate_keyword']]
    return d_ploc

def initial_library():
    global d_d, d_lev, d_loc, d_t, d_ploc, phrases_list, phrases_list_post, df,dc
    path = './libs/disease_lib.csv'
    d_d = pd.read_csv(path)

    path_lev = './libs/level_lib.csv'
    d_lev = pd.read_csv(path_lev)
    path_loc = './libs/location_lib.csv'
    d_loc = pd.read_csv(path_loc)
    path_t = './libs/type_lib.csv'
    d_t = pd.read_csv(path_t)
    path_ploc = './libs/postlocation_lib.csv'
    d_ploc = pd.read_csv(path_ploc)
    path_change = './libs/position_change.csv'
    dc = pd.read_csv(path_change)

    process_postlocation(d_ploc, dc)


    phrases_list = get_phrases_list(d_lev, 'level')
    phrases_list += get_phrases_list(d_loc, 'location')
    phrases_list_post = get_phrases_list(d_ploc, 'post_location')

    df = pd.DataFrame(columns=['id', 'report_name'])
    df['report_name'] = df['report_name'].astype(object)
    index = 0
    for i in range(len(d_d)):
        names = d_d.iloc[i]['report_name'].split(';')
        for name in names:
            df.loc[index] = [int(d_d.iloc[i]['id']), name]
            # df.at[index,'report_name'] = name
            index += 1

def test_extract_report(study_id, report_path):
    '''
    extract json KeyInfo data from the report
    '''
    initial_library()
    nlp = spacy.load("en_ner_bc5cdr_md")

    path_all = '../mimic_all.csv'
    df_all = pd.read_csv(path_all)
    subject_id = df_all[df_all['study_id']==study_id]['subject_id'].values[0]

    path = report_path
    fold1 = 'p' + str(subject_id)[:2]
    fold2 = 'p' + str(subject_id)
    file_name = 's%s.txt' % str(study_id)
    file_path = os.path.join(path, fold1, fold2, file_name)
    with open(file_path, 'r') as f:
        ori_text = f.read()
    lib = []
    if 'FINDINGS:' in ori_text:
        text = ori_text[ori_text.find('FINDINGS:'):]
    elif 'IMPRESSION:' in ori_text:
        text = ori_text[ori_text.find('IMPRESSION:'):]
    t = text
    t = t.replace('\n', ' ')
    lib = lib + t.split('.')

    print('report:',ori_text)

    out, no_out = find_general(lib, nlp, print_test=False, uniform_name=True, fixed_order=True)

def gen_disease_json(report_path, print_test = False, stop=False, save=True, uniform_name=True, fixed_order=True):
    '''
    this function is used to generate the keyInfo data for each report. The keyInfo data is then used to generate questions.
    '''
    print('Start KeyInfo data extraction')
    initial_library()
    nlp = spacy.load("en_ner_bc5cdr_md") if print_test else None

    p1 = os.listdir(report_path)
    final_diseases = []

    print('start')
    for fold1 in p1:
        print(fold1)
        if fold1[0] != 'p':
            continue
        path2 = os.path.join(report_path,fold1)
        p2 = os.listdir(path2)
        for i in tqdm(range(len(p2))):
            fold2 = p2[i]
            path3 = os.path.join(path2,fold2)
            files = os.listdir(path3)
            for file in files:
                with open(os.path.join(path3, file), 'r') as f:
                    record = {}
                    record['study_id'] = file[1:-4]
                    record['subject_id'] = fold2[1:]
                    t = file[:-4] + '\n'
                    text = f.read()
                    lib = []
                    if 'FINDINGS:' in text:
                        text = text[text.find('FINDINGS:'):]
                    elif 'IMPRESSION:'in text:
                        text = text[text.find('IMPRESSION:'):]
                    t += text
                    t = t.replace('\n', ' ')
                    lib = lib + t.split('.')

                    out, no_out = find_general(lib,nlp,print_test, uniform_name, fixed_order)
                    record['entity'] = out
                    record['no_entity'] = no_out
                    if print_test:
                        print('final out:',out)
                        print('final noout:',no_out)
                    final_diseases.append(record)

                # if stop:
                #     break
            # if stop:
            #     break
        if stop:
            break

    if save:
        disease_path = '../all_diseases.json'
        with open(disease_path,'w') as f:
            json.dump(final_diseases,f, indent=4)

def if_positive_entity(entity, text):
    # determine if the entity is negative
    negative_part = re.search('(.*) (without|no |clear of|r/o |rule out|less likely)(.*)', text, re.I)
    if negative_part:
        negative_part = negative_part.group(3)
        if entity in negative_part:
            return False

    # determine if the entity is positive
    if entity in text.split():
        return True
    else:
        return False

def find_keywords_in_report(keyword, report_path, background_words = None, no_keyword=None):
    '''
    importance score: the ratio of the number of times the keyword appears in the report to the number of the total reports.
    inference score: keyword_num( in the report with background words) / background_num
    correlation score: background_num( in the report with keywords)  / keyword_num
    '''


    path = report_path
    p1 = os.listdir(path)
    final_diseases = []


    background_num = 0
    keyword_num= 0
    inf_nume = 0
    cor_nume = 0
    total_num = 0
    for fold1 in p1:
        print(fold1)
        if fold1[0] != 'p':
            continue
        path2 = os.path.join(path,fold1)
        p2 = os.listdir(path2)
        for i in tqdm(range(len(p2))):
            fold2 = p2[i]
            path3 = os.path.join(path2,fold2)
            files = os.listdir(path3)
            for file in files:
                background_found = False
                keyword_found = False
                total_num += 1
                with open(os.path.join(path3, file), 'r') as f:
                    record = {}
                    record['study_id'] = file[1:-4]
                    record['subject_id'] = fold2[1:]
                    t = file[:-4] + '\n'
                    ori_text = f.read()
                    lib = []
                    if 'FINDINGS:' in ori_text:
                        text = ori_text[ori_text.find('FINDINGS:'):]
                    elif 'IMPRESSION:'in ori_text:
                        text = ori_text[ori_text.find('IMPRESSION:'):]
                    else:
                        text = ori_text
                    t += text
                    t = t.replace('\n', ' ')
                    lib = lib + t.split('.')


                    if background_words is not None:
                        for l in lib:
                            if any([if_positive_entity(b,l) for b in background_words]):
                                background_num += 1
                                background_found = True
                                break



                    for l in lib:
                        if any([if_positive_entity(k,l) for k in keyword]):
                            if no_keyword is not None:
                                if no_keyword in l:
                                    continue
                            keyword_num += 1
                            keyword_found = True
                            break

                    if background_found and keyword_found:
                        inf_nume += 1
                        cor_nume += 1

                    if background_found or keyword_found:
                        print('subject_id:', fold2[1:], 'study_id:', file[1:-4])
                        print(t)
                        print('importance score:%.4f'% (keyword_num/total_num))
                        if background_num != 0 and background_words is not None:
                            print("keyword inference score:%.4f"% (inf_nume/background_num))
                        if keyword_num != 0 and background_words is not None:
                            print("keyword correlation score:%.4f"% (cor_nume/keyword_num))
                        print('\n\n\n')


def post_process_record(out, no_out,record):
    record['entity'] = out
    record['no_entity'] = no_out
    return record



def create_question_set():
    dict = {}
    dict['abnormality'] = ['what abnormalities are seen in this image?', 'what abnormalities are seen in the xxx?','is there evidence of any abnormalities in this image?']
    dict['presence'] = ['is there evidence of xxx in this image?', 'is there xxx?', 'is there xxx in the lxxx?']
    dict['view'] = ['which view is this image taken?', 'is this PA view?', 'is this AP view?']
    dict['location'] = ['where in the image is the xxx located?', 'where is the xxx?', 'is the xxx located on the left side or right side?']
    dict['level'] = ['what level is the xxx?']
    dict['type'] = ['what type is the xxx?']
    dict['difference'] = ['what has changed compared to the reference image?', 'what has changed in the lxxx area?']
    return dict

def abnormality_ques(record, less_yes_no):
    while 1:
        q_id = np.random.randint(len(question_set['abnormality']))
        if q_id >= 2:
            if less_yes_no:
                if np.random.rand() < 0.9: # keep 10% of questions of yes/no
                    return None
        if q_id == 0:
            if len(record['entity']) == 0:
                if np.random.rand(1)[0]>0.5:
                    return None
                else:
                    continue
            answer = []
            for key in record['entity']:
                answer.append(key)
            answer = ', '.join(answer)
            question = question_set['abnormality'][q_id]
            return (question, answer)
        elif q_id == 1:
            if len(record['entity']) == 0:
                if np.random.rand(1)[0] > 0.5:
                    return None
                else:
                    continue
            entities = record['entity'].copy()
            # random shuffle the entities
            keys = list(entities.keys())
            random.shuffle(keys)
            entities = {key: entities[key] for key in keys}

            for j in range(len(entities)):
                entity = entities.popitem()
                if entity[1]['post_location'] is not None:
                    question = question_set['abnormality'][q_id].replace('the xxx', entity[1]['post_location'])
                    answer = entity[0]
                    return (question, answer)
            continue
        elif q_id == 2:
            #'is there evidence of any abnormalities in this image?'
            question = question_set['abnormality'][q_id]
            if len(record['entity'])>0:
                answer = 'yes'
            else:
                answer = 'no'
            return (question, answer)
        elif q_id == 3:
            # 'is this image normal?'
            question = question_set['abnormality'][q_id]
            if len(record['entity']) > 0:
                answer = 'no'
            else:
                answer = 'yes'
            return (question, answer)
        elif q_id == 4:
            continue

def get_exist_disease_id(record):
    id_set = set()
    for key in record['entity']:
        id = df[df['report_name']==key]['id'].values
        id_set.add(id[0])
    return id_set

def pres_ques0_no(record,question):
    # use no_entity to answer this question
    n_id = np.random.randint(len(record['no_entity']))
    no_entity = record['no_entity'][n_id]
    question = question.replace('xxx', no_entity)
    answer = 'no'
    return (question, answer)
def pres_ques0_yes(record,question):
    # use "entity" to answer this question. but prefer the answer to be NO. so randomly select from all the names
    n_id = np.random.randint(len(d_d))
    disease_id = d_d.iloc[n_id]['id']
    disease_name = d_d.iloc[n_id]['official_name']
    question = question.replace('xxx', disease_name)
    exist_id = get_exist_disease_id(record)
    if disease_id in exist_id:
        answer = 'yes'
    else:
        answer = 'no'
    return (question, answer)


def presence_ques(record, less_yes_no):
    if less_yes_no:
        random_num = np.random.rand(1)[0]
        if random_num < 0.9: # keep 10% of questions of yes/no
            return None
    if len(record['entity']) == 0 and len(record['no_entity']) == 0:
        return None
    while 1:
        q_id = np.random.randint(len(question_set['presence']))
        if q_id == 0 or q_id == 1:
            #'is there evidence of xxx in this image?'
            question = question_set['presence'][q_id]
            if np.random.rand(1)[0] > 0.5: # no
                if np.random.rand(1)[0] > 0.5:
                    if len(record['no_entity'])>0:
                        return pres_ques0_no(record, question)
                    else:
                        return pres_ques0_yes(record, question)
                else:
                    if len(record['entity'])> 0:
                        return pres_ques0_yes(record, question)
                    else:
                        return pres_ques0_no(record, question)

            else: # yes
                if len(record['entity'])>0:
                    entities = record['entity'].copy()
                    # random shuffle the entities
                    keys = list(entities.keys())
                    random.shuffle(keys)
                    entities = {key: entities[key] for key in keys}

                    disease_name = entities.popitem()[0]
                    question = question.replace('xxx', disease_name)
                    answer = 'yes'
                    return (question, answer)
                else:
                    continue
        elif q_id == 2:
            #'is there xxx in the lxxx?'
            question = question_set['presence'][q_id]
            returned = sub_ques_pres_loc(record, question)
            if returned is not None:
                return returned
            else:
                continue

def sub_ques_pres_loc(record, question):
    entities = record['entity'].copy()
    # random shuffle the entities
    keys = list(entities.keys())
    random.shuffle(keys)
    entities = {key: entities[key] for key in keys}

    if np.random.rand(1)[0] > 0.5:  # yes
        if len(record['entity']) > 0:
            for j in range(len(entities)):
                item = entities.popitem()
                disease_name = item[0]
                question = question.replace(' xxx', ' '+disease_name)
                if item[1]['location'] is not None:
                    location = ' '.join(item[1]['location'])
                    if location == 'left' or location == 'right':
                        location = location + ' lung'
                    if location == 'bilateral':
                        question = 'is the ' + disease_name + ' bilateral?'
                    else:
                        location = location + ' area'
                        question = question.replace('lxxx', location)
                    answer = 'yes'
                    return (question, answer)
                elif item[1]['post_location'] is not None:
                    location = item[1]['post_location']
                    question = question.replace('the lxxx', location)
                    answer = 'yes'
                    return (question, answer)
            return None
        else:
            return None
    else:  # no
        if len(record['entity']) > 0:
            for entity in entities:
                if entities[entity]['post_location'] is not None:
                    keywords = \
                    d_ploc[d_ploc['post_location'] == entities[entity]['post_location']]['relate_keyword'].values[
                        0].split(';')
                    while 1:
                        location = random.choice(d_ploc['post_location'].values)
                        if not check_any_in(keywords, location):
                            break # find a location that keyword is not in
                        # if keyword not in location:
                        #     break
                    question = question.replace(' xxx', ' '+entity)
                    question = question.replace('the lxxx', location)
                    answer = 'no'
                    return (question, answer)
            # all post_location is None
            # consider_pre
            for j in range(len(entities)):
                item = entities.popitem()
                disease_name = item[0]
                this_question = question.replace(' xxx', ' ' + disease_name)
                if item[1]['location'] is not None:
                    location = ' '.join(item[1]['location'])
                    if 'left' in location:
                        location = location.replace('left', 'right')
                    elif 'right' in location:
                        location = location.replace('right', 'left')
                    elif 'mid to lower' in location:
                        location = 'upper to mid'
                    elif 'upper to mid' in location:
                        location = 'mid to lower'
                    elif 'upper' in location:
                        location = location.replace('upper', 'lower')
                    elif 'lower' in location:
                        location = location.replace('lower', 'upper')
                    elif 'middle' in location or 'mid' in location:
                        location = location.replace('middle', random.choice(['upper', 'lower'])).replace('mid',random.choice(['upper','lower']))
                    else:
                        continue
                    location = location + ' area'
                    this_question = this_question.replace('lxxx', location)
                    answer = 'no'
                    return (this_question, answer)
            # all none
            return None

def view_ques(record, less_yes_no):
    study_id = record['study_id']
    subject_id = record['subject_id']
    try:
        view = d_all[d_all['study_id'] == int(study_id)]['view'].values[0]
    except:
        return None
    if view == 'antero-posterior':
        view = 'AP view'
        if np.random.rand(1)[0]>0.05:
            return None
    elif view == 'postero-anterior':
        view = 'PA view'
    else:
        return None
    if np.random.rand(1)[0]>0.5:
        q_id = 0
        question = question_set['view'][q_id]
        answer = view
        return (question, answer)
    else:
        if less_yes_no:
            if np.random.rand(1)[0] < 0.9:
                return None
            return None
        if np.random.rand(1)[0]>0.5:
            q_id = 1
            if view == 'PA view':
                answer = 'yes'
            else:
                answer = 'no'
        else:
            q_id = 2
            if view == 'AP view':
                answer = 'yes'
            else:
                answer = 'no'
        question = question_set['view'][q_id]
        return (question, answer)

def location_ques(record):
    q_id = np.random.randint(len(question_set['location']))
    entities = record['entity'].copy()
    # random shuffle the entities
    keys = list(entities.keys())
    random.shuffle(keys)
    entities = {key: entities[key] for key in keys}

    if q_id == 0 or q_id == 1:
        question = question_set['location'][q_id]
        for i in range(len(entities)):
            entity = entities.popitem()
            if entity[1]['location'] is not None:
                if 'left' in entity[1]['location'] and 'right' in entity[1]['location']: # left and right # todo: now considered two findings of the same disease, in outer loop. But inner loop has also been added the capability of detecting multiple(two) findings in one sentences.
                    continue
                # if entity[1]['location2'] is not None or entity[1]['type2'] is not None or entity[1]['level2'] is not None or entity[1]['post_location2'] is not None:
                #     continue
                name = entity[1]['entity_name']
                question = question.replace('xxx', name)
                answer = ' '.join(entity[1]['location'])
                answer += ' area'
                if entity[1]['location2'] is not None:
                    if 'left' in entity[1]['location2'] and 'right' in entity[1]['location2']:
                        continue
                    answer += ' and ' + ' '.join(entity[1]['location2'])
                    answer += ' area'
                return (question, answer)
            if entity[1]['post_location'] is not None:
                if 'left' in entity[1]['post_location'] and 'right' in entity[1]['post_location']: # left and right
                    continue
                if entity[1]['location2'] is not None or entity[1]['type2'] is not None or entity[1]['level2'] is not None or entity[1]['post_location2'] is not None:
                    continue
                name = entity[1]['entity_name']
                question = question.replace('xxx', name)
                answer = entity[1]['post_location']
                return (question, answer)
        return None
    elif q_id ==2:
        question = question_set['location'][q_id]
        for i in range(len(entities)):
            entity = entities.popitem()
            if entity[1]['location'] is not None:
                if 'left' in entity[1]['location'] and 'right' in entity[1]['location']: # left and right
                    continue
                if entity[1]['location2'] is not None or entity[1]['type2'] is not None or entity[1]['level2'] is not None or entity[1]['post_location2'] is not None:
                    continue
                question = question.replace('xxx', entity[1]['entity_name'])
                if 'left' in entity[1]['location']:
                    answer = 'left side'
                    return (question, answer)
                elif'right' in entity[1]['location']:
                    answer = 'right side'
                    return (question, answer)
            if entity[1]['post_location'] is not None:
                if 'left' in entity[1]['post_location'] and 'right' in entity[1]['post_location']: # left and right
                    continue
                if entity[1]['location2'] is not None or entity[1]['type2'] is not None or entity[1]['level2'] is not None or entity[1]['post_location2'] is not None:
                    continue
                question = question.replace('xxx', entity[1]['entity_name'])
                if 'left' in entity[1]['post_location']:
                    answer = 'left side'
                    return (question, answer)
                elif 'right' in entity[1]['post_location']:
                    answer = 'right side'
                    return (question, answer)
        return None
    elif q_id == 3: # not in use
        question = question_set['location'][q_id]
        return sub_ques_pres_loc(record, question)

def level_ques(record):
    question = question_set['level'][0]
    entities = record['entity'].copy()
    # random shuffle the entities
    keys = list(entities.keys())
    random.shuffle(keys)
    entities = {key: entities[key] for key in keys}

    for i in range(len(record['entity'])):
        entity = entities.popitem()
        if entity[1]['level'] is not None:
            question = question.replace('xxx',entity[1]['entity_name'] )
            answer = ' '.join(entity[1]['level'])
            return (question, answer)
    return None

def type_ques(record):
    question = question_set['type'][0]
    entities = record['entity'].copy()
    # random shuffle the entities
    keys = list(entities.keys())
    random.shuffle(keys)
    entities = {key: entities[key] for key in keys}

    for i in range(len(record['entity'])):
        entity = entities.popitem()
        if entity[1]['type'] is not None:
            question = question.replace('xxx',entity[1]['entity_name'] )
            answer = ' '.join(entity[1]['type'])
            return (question, answer)
    return None

def convert_list_of_name2offical(list):
    for i in range(len(list)):
        try:
            name = d_d[d_d['report_name'].str.contains(list[i])]['official_name'].values[0]
        except:
            continue
        list[i] = name
    return list

def get_caption(adding, dropping,):
    if len(adding) == 1:
        output1 = 'the main image has an additional finding of'
    elif len(adding) > 1:
        output1 = 'the main image has additional findings of'
    elif len(adding) == 0:
        output1 = ''
    for item in adding:
        if item == adding[-1] and len(adding) != 1:
            output1 = output1 + ' and ' + item
        else:
            if len(adding) == 1:
                output1 = output1 + ' ' + item
            else:
                output1 = output1 + ' ' + item + ','
    if len(adding) != 0:
        output1 = output1 + ' than the reference image. '

    if len(dropping) == 1:
        output2 = 'the main image is missing the finding of'
    elif len(dropping) > 1:
        output2 = 'the main image is missing the findings of'
    elif len(dropping) == 0:
        output2 = ''
    for item in dropping:
        if item == dropping[-1] and len(dropping) != 1:
            output2 = output2 + ' and ' + item
        else:
            if len(dropping) == 1:
                output2 = output2 + ' ' + item
            else:
                output2 = output2 + ' ' + item + ','

    if len(dropping) != 0:
        output2 = output2 + ' than the reference image. '
    return output1 + output2

def diff_ques(record, ref_record):
    question = question_set['difference'][0]
    entities = record['entity'].copy()
    # random shuffle the entities
    keys = list(entities.keys())
    random.shuffle(keys)
    entities = {key: entities[key] for key in keys}

    entities_ref = ref_record['entity'].iloc[0].copy()
    answer = ''
    finding_name = convert_list_of_name2offical(list(entities))
    ref_finding_name = convert_list_of_name2offical(list(entities_ref))
    for i, key in enumerate(entities):
        for j, key_ref in enumerate(entities_ref):
            # if df[df['report_name']==key]['id'].values[0] == df[df['report_name']==key_ref]['id'].values[0]:
            if finding_name[i] == ref_finding_name[j]:
                if entities[key]['level'] != entities_ref[key_ref]['level'] and entities[key]['level'] is not None and entities_ref[key_ref]['level'] is not None:
                    if entities[key]['location'] is not None and entities_ref[key_ref]['location'] is not None:
                        common_area = list(set(entities[key]['location']).intersection(entities_ref[key_ref]['location']))
                        if common_area != [] and ' '.join(common_area) != 'bilateral':
                            if ' '.join(common_area) == 'left' or ' '.join(common_area) == 'right':
                                common_area = [' '.join(common_area) + ' lung']
                            question = 'what has changed in the %s area?'% ' '.join(common_area)
                            name = finding_name[i]
                            answer = 'the level of %s has changed from %s to %s. '% (name, ' '.join(entities_ref[key_ref]['level']), ' '.join(entities[key]['level']))
                            return (question, answer)
                    # name = d_d[d_d['report_name'].str.contains(key)]['official_name'].values[0]
                    name = finding_name[i]
                    answer += 'the level of %s has changed from %s to %s. '% (name, ' '.join(entities_ref[key_ref]['level']), ' '.join(entities[key]['level']))
    adding = []
    for name in finding_name:
        if name not in ref_finding_name:
            if name == 'pneumonia' and 'infection' in ref_finding_name or name == 'infection' and 'pneumonia' in ref_finding_name:
                continue
            adding.append(name)
    missing = []
    for name in ref_finding_name:
        if name not in finding_name:
            if name == 'pneumonia' and 'infection' in ref_finding_name or name == 'infection' and 'pneumonia' in ref_finding_name:
                continue
            missing.append(name)
    if adding != [] or missing != []:
        caption = get_caption(adding, missing)
        answer += caption
    if answer == '':
        answer = 'nothing has changed.'



    return (question, answer)


def initial_question_record(record):
    question_record = {}
    question_record['study_id'] = record['study_id']
    question_record['subject_id'] = record['subject_id']
    question_record['ref_id'] = None
    question_record['question_type'] = None
    question_record['question'] = None
    question_record['answer'] = None
    return question_record

def get_all_types_of_question(i, diseases_json, question_set_of_this_record,question_record, ref_id, ref_record, pair_questions, less_yes_no):
    # abnormality:
    qa_pair = abnormality_ques(diseases_json[i],less_yes_no)
    if qa_pair is not None and qa_pair[0] not in question_set_of_this_record:
        question_set_of_this_record.add(qa_pair[0])
        question_record['question'] = qa_pair[0]
        question_record['answer'] = qa_pair[1]
        question_record['question_type'] = 'abnormality'
        question_record['ref_id'] = ref_id
        pair_questions.append(question_record.copy())
        question_record = initial_question_record(diseases_json[i])
    # presence
    qa_pair = presence_ques(diseases_json[i], less_yes_no)
    if qa_pair is not None and qa_pair[0] not in question_set_of_this_record:
        question_set_of_this_record.add(qa_pair[0])
        question_record['question'] = qa_pair[0]
        question_record['answer'] = qa_pair[1]
        question_record['question_type'] = 'presence'
        question_record['ref_id'] = ref_id
        pair_questions.append(question_record.copy())
        question_record = initial_question_record(diseases_json[i])
    # view
    qa_pair = view_ques(diseases_json[i], less_yes_no)
    if qa_pair is not None and qa_pair[0] not in question_set_of_this_record:
        question_set_of_this_record.add(qa_pair[0])
        question_record['question'] = qa_pair[0]
        question_record['answer'] = qa_pair[1]
        question_record['question_type'] = 'view'
        question_record['ref_id'] = ref_id
        pair_questions.append(question_record.copy())
        question_record = initial_question_record(diseases_json[i])
    # location
    qa_pair = location_ques(diseases_json[i])
    if qa_pair is not None and qa_pair[0] not in question_set_of_this_record:
        question_set_of_this_record.add(qa_pair[0])
        question_record['question'] = qa_pair[0]
        question_record['answer'] = qa_pair[1]
        question_record['question_type'] = 'location'
        question_record['ref_id'] = ref_id
        pair_questions.append(question_record.copy())
        question_record = initial_question_record(diseases_json[i])
    # level
    qa_pair = level_ques(diseases_json[i])
    if qa_pair is not None and qa_pair[0] not in question_set_of_this_record:
        question_set_of_this_record.add(qa_pair[0])
        question_record['question'] = qa_pair[0]
        question_record['answer'] = qa_pair[1]
        question_record['question_type'] = 'level'
        question_record['ref_id'] = ref_id
        pair_questions.append(question_record.copy())
        question_record = initial_question_record(diseases_json[i])
    # type
    qa_pair = type_ques(diseases_json[i])
    if qa_pair is not None and qa_pair[0] not in question_set_of_this_record:
        question_set_of_this_record.add(qa_pair[0])
        question_record['question'] = qa_pair[0]
        question_record['answer'] = qa_pair[1]
        question_record['question_type'] = 'type'
        question_record['ref_id'] = ref_id
        pair_questions.append(question_record.copy())
        question_record = initial_question_record(diseases_json[i])
    # difference
    qa_pair = diff_ques(diseases_json[i], ref_record)
    if qa_pair is not None and qa_pair[0] not in question_set_of_this_record:
        question_set_of_this_record.add(qa_pair[0])
        question_record['question'] = qa_pair[0]
        question_record['answer'] = qa_pair[1]
        question_record['question_type'] = 'difference'
        question_record['ref_id'] = ref_id
        pair_questions.append(question_record.copy())

    return question_set_of_this_record, pair_questions

def question_gen(less_yes_no=False):
    '''
    :param less_yes_no:  if generating less yes/no type questions
    :param small_sample: this parameter is for testing a small portion of data
    :return:
    '''
    if not os.path.exists('temp'):
        os.makedirs('temp')
    print('Start question generation')
    initial_library()
    path = '../all_diseases.json'
    global question_set
    question_set = create_question_set()
    with open(path, 'r') as f:
        diseases_json = json.load(f)
    path_all = '../mimic_all.csv'
    global d_all
    d_all = pd.read_csv(path_all)

    diseases_df = pd.DataFrame(diseases_json)
    pair_questions = []
    repeat_pair = 1
    repeat_ques_gen = 1
    ran = range(len(diseases_json))
    for i in tqdm(ran):
        try:
            view = d_all[d_all['study_id'] == int(diseases_json[i]['study_id'])]['view'].values[0]
        except:
            continue
        if not (view == 'antero-posterior' or view == 'postero-anterior'):
            continue
        tried_ref_id_set = set()
        tried_ref_id_set.add(diseases_json[i]['study_id'])
        for j in range(repeat_pair): # each main image has 3(repeat time) groups(pair images) of questions.
            question_set_of_this_record = set()
            question_record = initial_question_record(diseases_json[i])

            ref_id_candidate = list(diseases_df[diseases_df['subject_id']==diseases_json[i]['subject_id']]['study_id'].values)
            random.shuffle(ref_id_candidate)
            while len(tried_ref_id_set) < len(diseases_df[diseases_df['subject_id'] == diseases_json[i]['subject_id']]['study_id'].values) :
                # ref_id means reference study id
                ref_id = ref_id_candidate.pop()
                try:
                    view = d_all[d_all['study_id'] == int(ref_id)]['view'].values[0]
                except:
                    tried_ref_id_set.add(ref_id)
                if ref_id not in tried_ref_id_set and (view == 'antero-posterior' or view == 'postero-anterior'):
                    tried_ref_id_set.add(ref_id)
                    break
                tried_ref_id_set.add(ref_id)
            if len(tried_ref_id_set) == len(
                    diseases_df[diseases_df['subject_id'] == diseases_json[i]['subject_id']]['study_id'].values):
                continue

            ref_record = dict(diseases_df[diseases_df['study_id'] == ref_id])

            for k in range(repeat_ques_gen):
                question_set_of_this_record, pair_questions = get_all_types_of_question(i, diseases_json, question_set_of_this_record,question_record, ref_id, ref_record, pair_questions, less_yes_no)

    pd_pair_questions = pd.DataFrame(pair_questions)
    name = 'temp/mimic_pair_questions_temp.csv'
    pd_pair_questions.to_csv(name, index=False)

def adding(pair, x):
    return pair + x




def statistic():
    path = 'datasets/mimic_pair_questions.csv'
    d = pd.read_csv(path)
    print('len',len(set(d['subject_id'])))
    types = ['abnormality','presence','view','location','level','type','difference']
    print('all question answer pairs', len(d))

    yes_num = 0
    no_num = 0
    for t in types:
        total = len((d[d['question_type'] == t]))
        print('%s, %d, %.2f%%'%(t, total, total/len(d)*100))
        closed = len(d[(d['question_type'] == t) & ((d['answer'] == 'no') | (d['answer'] == 'yes')) ])
        yes_num = len(d[(d['question_type'] == t) & (d['answer'] == 'yes')])
        no_num = len(d[(d['question_type'] == t) & (d['answer'] == 'no')])
        print('open, %d, %.2f%%'%(total-closed, (total-closed)/total*100))
        print('closed, %d, %.2f%%'%(closed, (closed)/total*100))
        print('yes, %d, %.2f%%'%(yes_num, yes_num/len(d)*100))
        print('no, %d, %.2f%%'%(no_num, no_num/len(d)*100))
        print('')


    img_pair_num = 0
    img_pair_set = set()
    img_set = set()
    img_num = 0
    for i in tqdm(range(len(d))):
        if d['study_id'].values[i] not in img_set:
            img_set.add(d['study_id'].values[i])
            img_num += 1
        if d['ref_id'].values[i] not in img_set:
            img_set.add(d['ref_id'].values[i])
            img_num += 1
        pair = (d['study_id'].values[i], d['ref_id'].values[i])
        if pair not in img_pair_set:
            img_pair_set.add(pair)
            img_pair_num += 1
    print('img_pair_num', img_pair_num)
    print('img_num', img_num)

    answer_set = set(d[d["question_type"] != 'difference']['answer'].values)
    print('answer set length', len(answer_set))


def transform_pos_tag(tag_list, d_pos, max_seq):
    out = []
    for item in tag_list:
        tag = item[1]
        id = d_pos[d_pos['tag'] == tag]['id'].values[0]
        out.append(id)
    for i in range(len(out),max_seq):
        out.append(0)
    return out


def process(list, diseases_list, mode = 'strict'):
    out = []
    for i in range(len(list)):
        if mode == 'strict':
            if list[i] == 1:
                out.append(diseases_list[i].lower())
        else:
            if list[i] == 1 or list[i] == -1:
                out.append(diseases_list[i].lower())
    return out


def find_in_testset():
    # for generating report by liangchen lilu
    path = '../test_set_pidsid.csv'
    df = pd.read_csv(path)
    # l = []
    ids = set()
    for i in tqdm(range(len(df))):
        id = df.iloc[i]['1'][1:]
        ids.add(id)
    # print(ids)
    return ids

def contains_number(string):
    return any(char.isdigit() for char in string)

def are_capitals(string):
    for char in string:
        if char.isalpha() and not char.isupper():
            return False
    return True

def find_section_words(report_path):
    # this way has been approved not good. abandoned
    path_all = '../mimic_all.csv'
    df_all = pd.read_csv(path_all)
    study_ids = df_all['study_id'].values

    lib = set()
    for study_id in tqdm(study_ids):
        subject_id = df_all[df_all['study_id'] == study_id]['subject_id'].values[0]

        path = report_path
        fold1 = 'p' + str(subject_id)[:2]
        fold2 = 'p' + str(subject_id)
        file_name = 's%s.txt' % str(study_id)
        file_path = os.path.join(path, fold1, fold2, file_name)
        with open(file_path, 'r') as f:
            ori_text = f.read()
        # if 'FINDINGS:' in ori_text:
        #     text = ori_text[ori_text.find('FINDINGS:'):]
        # elif 'IMPRESSION:' in ori_text:
        #     text = ori_text[ori_text.find('IMPRESSION:'):]
        ts = ori_text.split('\n')
        for t in ts:
            t = t.strip()
            if ':' in t:
                word = t[:t.find(':')]
                if word not in lib:
                    if contains_number(word) or not are_capitals(word):
                        continue
                    print('report for %s:'%(word[:-1]), ori_text)
                    lib.add(word)
                # if word[-1:] == ':':
                #     if word[:-1] not in lib:
                #         lib.add(word[:-1])


        # print('report:', ori_text)

def find_best_paragraph(report_path, study_ids = None):
    path_all = '../mimic_all.csv'
    df_all = pd.read_csv(path_all)
    if study_ids is None:
        study_ids = df_all['study_id'].values


    forbidden_words = ['WET READ', 'INDICATION','EXAM','COMPARISON', 'HISTORY']

    for study_id in tqdm(study_ids):
        subject_id = df_all[df_all['study_id'] == study_id]['subject_id'].values[0]

        path = report_path
        fold1 = 'p' + str(subject_id)[:2]
        fold2 = 'p' + str(subject_id)
        file_name = 's%s.txt' % str(study_id)
        file_path = os.path.join(path, fold1, fold2, file_name)
        with open(file_path, 'r') as f:
            ori_text = f.read()

        print('\nstudy_id:', study_id)
        # 1. find FINDINGS if it exists
        if 'FINDINGS:' in ori_text:
            text = ori_text[ori_text.find('FINDINGS:'):]
            print('report:', text)
            print('==============')
        else:
            # 2. find the longest paragraph. First, sort it by length
            paragraphs = ori_text.replace('\n \n','\n\n').split('\n\n')
            paragraphs = sorted(paragraphs, key=len)

            while len(paragraphs)>1:
                # 3. rule out the paragraph with forbidden words
                if check_any_in(forbidden_words,paragraphs[-1]):
                    paragraphs.pop()
                    continue
                output_report = paragraphs[-1]
                # 4. add IMPRESSION if it exists
                if 'IMPRESSION' in ori_text and 'IMPRESSION' not in paragraphs[-1]:
                    impression = ori_text[ori_text.find('IMPRESSION:'):]
                    if output_report in impression:
                        output_report = impression
                    elif impression in output_report:
                        pass
                    else:
                        output_report = output_report + '\n' + impression


                print('report:', output_report)
                print('==============')
                print('original report:', ori_text)
                print('==============')

                break
            if len(paragraphs) <= 1:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--extract_json", action='store_true',  help="extract KeyInfo json file")
    parser.add_argument("-q", "--gen_question", action='store_true', help="generate question-asnwer pairs")
    parser.add_argument("-r", "--report_path", type=str, default='/home/xinyue/dataset/mimic_reports', help="path to the report folder")
    args = parser.parse_args()
    ### EXECUTING PART ####
    if not args.extract_json and not args.gen_question:
        print('please choose one option. either -j for json extraction or -q for question generation')
        exit()
    if args.extract_json:
        gen_disease_json(report_path=args.report_path, save=True)  # generate keyInfo data
    if args.gen_question:
        question_gen()  # generate question csv

    # statistic()
    print('finished generate dataset')

if __name__=='__main__':
    main()





