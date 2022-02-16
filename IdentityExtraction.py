import json
import pandas as pd
import nltk
import spacy
import os
import textacy
import textacy.spacier.utils as spacy_utils
import re
import itertools

def check_whether_empty(s):
    if not s[0][0]:
        return 'Empty'
    else:
        return 'Not Empty'

def get_first_sentence(s):
    s = s[0][0]
    #print(s)
    s = nltk.sent_tokenize(s)[0]
    #s = re.sub('\.(?!(\S[^. ])|\d)', '', s, count = 1)
    #print(s)
    #s = re.sub(r'\.', '', s, count = 1 )
    s = re.sub("([\(\[]).*?([\)\]])", "", s)
    #print(s)
    return s

def get_subj_obj_verb(doc):
    for sent in doc.sents:
        verbs = [tok for tok in sent if tok.pos_ == 'VERB' or tok.pos_ == 'AUX']
        #print(verbs)
        start_i = sent[0].i
        for verb in verbs:
            subjs = spacy_utils.get_subjects_of_verb(verb)
            #print(subjs)
            if not subjs:
                continue
            objs = spacy_utils.get_objects_of_verb(verb)
            #print(objs)
            if not objs:
                continue

            # add adjacent auxiliaries to verbs, for context
            # and add compounds to compound nouns
            verb_span = spacy_utils.get_span_for_verb_auxiliaries(verb)
            verb = sent[verb_span[0] - start_i : verb_span[1] - start_i + 1]
            for subj in subjs:
                subj = sent[
                    spacy_utils.get_span_for_compound_noun(subj)[0]
                    - start_i : subj.i
                    - start_i
                    + 1
                    ]
                #print(subj)
                for obj in objs:
                    if obj.pos_ == 'NOUN':
                        span = spacy_utils.get_span_for_compound_noun(obj)
                    elif obj.pos_ == 'VERB':
                        span = spacy_utils.get_span_for_verb_auxiliaries(obj)
                    else:
                        span = (obj.i, obj.i)
                    obj = sent[span[0] - start_i : span[1] - start_i + 1]
                    #print(obj)
                    #print(obj
                    #print(subj," , ", verb," , ",obj)
                    yield (subj.text, verb.text, obj.text)
                    #print(subj," , ", verb," , ",obj)

def get_identities(path):
    files = os.listdir(path)
    nlp = spacy.load('en_core_web_md')
    for file in files:
        print(file)
        records = map(json.loads, open(os.path.join(path, file) , encoding = 'utf8'))
        df = pd.DataFrame.from_records(records)
        df['state'] = df['context'].apply(lambda x: check_whether_empty(x))
        df = df[df['state'] != 'Empty'].reset_index(drop = True)
        df['context'] = df['context'].apply(lambda x: get_first_sentence(x))
        sents = df['context'].to_list()
        docs = list(nlp.pipe(sents))
        l = []
        #df1 = pd.DataFrame()
        d = {}
        for i, doc in enumerate(docs):
            list_names = list(get_subj_obj_verb(doc))
            if len(list_names) > 1:
                d['name_searched'] = df.iloc[i].name_searched
                d['identities'] = [list_names[i][-1] for i in range(len(list_names))]
                #df1.append(d, ignore_index = False)
                l.append(d)
            elif len(list_names) == 1:
                d['name_searched'] = df.iloc[i].name_searched
                d['identities'] = [list_names[0][-1]]
                #df1.append(d, ignore_index = False)
                l.append(d)
            else:
                d['name_searched'] = df.iloc[i].name_searched
                d['identities'] = []
                #df1.append(d, ignore_index = False)
                l.append(d)
            d = {}
        #df1.to_csv('{}_name_identities.csv'.format(file[0:file.index('.json')]))

        with open('{}_json_identities.json'.format(file[0:file.index('.ndjson')]), 'w') as json_file:
            json.dump(l, json_file)
        
def main_func():
    l = []
    path = 'E:/ResearchWork/CurrentResearch/output'
    get_identities(path)

if __name__=='__main__':

    main_func()
    









