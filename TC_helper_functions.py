import os
import string
import re
import pickle

import pandas as pd
import numpy as np
from datetime import datetime
from langdetect import detect

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
# Voikko for Windows
from libvoikko import Voikko
Voikko.setLibrarySearchPath("Voikko")

# Initiate Voikko Class right away
v = Voikko(u"fi")

def load_dataset_improved(data_raw_dir):
    all_files = [filename for filename in os.listdir(data_raw_dir) if filename.endswith(".csv")]
    df_original = pd.concat((pd.read_csv(os.path.join(data_raw_dir, f)) for f in all_files))
    print("Number of tickets before any filtering: {}.".format(len(df_original)))
    return df_original

# Määrittele ja lataa stopwordsit
def load_stopwords(path, general=True, it=True, names=True):
    
    # General stopwords
    if general:
        stopwords_en = set(pd.read_csv(os.path.join(path, "stopwords_en.csv"), header=None)[0])
        stopwords_fi = set(pd.read_csv(os.path.join(path, "stopwords_fi.csv"), header=None)[0])

        all_stopwords_set = stopwords_en | stopwords_fi
    else:
        all_stopwords_set = set()
    
    # Efima & IT stopwords
    if it:
        all_efima_set = set(pd.read_csv(os.path.join(path, "stopwords_efima.csv"), header=None)[0])
    else:
        all_efima_set = set()
    
    # Finnish name stopwords
    if names:
        first_malename_list = pd.read_csv(os.path.join(path, "etunimet_miehet.csv"))
        first_femalename_list = pd.read_csv(os.path.join(path, "etunimet_naiset.csv"))
        family_malename_list = pd.read_csv(os.path.join(path, "sukunimet.csv"))

        first_malename_set = set(first_malename_list["Etunimi"])
        first_femalename_set = set(first_femalename_list["Etunimi"])
        family_malename_set = set(family_malename_list["Sukunimi"])

        all_name_set = first_malename_set | first_femalename_set | family_malename_set

        names_to_include_in_analysis = ["Aili"]
        all_name_set.remove(names_to_include_in_analysis[0])
        
        all_names_set = set([val.lower() for val in all_name_set])
    else:
        all_names_set = set()
        
    all_stopwords = all_stopwords_set | all_efima_set | all_names_set
    
    return all_stopwords

def num_there(s):
    return any(i.isdigit() for i in s)

def voikkonizer(words):
    voikko_processed = []
    for word in words:
        voikko_output = v.analyze(word)
        if  voikko_output == []:
            voikko_processed.append(word)
        else:
            voikko_processed_w = voikko_output[0]["BASEFORM"]
            voikko_processed.append(voikko_processed_w.lower())

    # Remove single chars
    voikko_processed = [val for val in voikko_processed if len(val) != 1]
    return voikko_processed
    
def nlp_preprocess(sentence):
    
    if type(sentence) is float:
        return ["nan"]
    
    pattern = r'\{.*?\}|\[.*?\]|\n|\xa0'
    pre_filt = re.sub(pattern, '', sentence)

    url_filt = re.sub(r'http\S+', '', pre_filt)
    email_filt = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', url_filt)

    punctuations_filt = email_filt.translate(str.maketrans('', '', string.punctuation))
    whitespace_filt = " ".join(punctuations_filt.split())
    
    footer_filt = re.sub('This e-mail.*?cooperation','', whitespace_filt, flags=re.DOTALL)
    footer_filt_1 = re.sub('The information.*?your computer','', footer_filt, flags=re.DOTALL)
    footer_filt_2 = re.sub('image.*?pngthumbnail','', footer_filt_1, flags=re.DOTALL)
    footer_filt_3 = re.sub('image.*?gif','', footer_filt_2, flags=re.DOTALL)
    footer_filt_4 = re.sub('image.*?jpgthumbnail','', footer_filt_3, flags=re.DOTALL)
    
    tokenization_filt = [w.lower() for w in footer_filt_4.split(" ") if not w.isdigit()]
    tokenization_filt_1 = [w for w in tokenization_filt if not num_there(w)]
    
    processed_words = voikkonizer(tokenization_filt_1)
    return processed_words

def get_nlp_processed_text(x, y):
    return nlp_preprocess(x) + nlp_preprocess(y)

def remove_stopwords(tokens, sw):
    filtered_sentence = [w for w in tokens if w not in sw]
    return filtered_sentence

def detect_language(sentence):
    if detect(sentence) == "fi":
        kieli = 'fi'
    else:
        kieli = 'other'
    return kieli

def compute_word_freq(corpus):
    vectorizer = CountVectorizer(min_df=5, lowercase=False, ngram_range=(1, 2))
    doc_term_matrix = vectorizer.fit_transform(corpus) 
    
    ngram_freqs = np.asarray(doc_term_matrix.sum(axis=0))[0]
    norm_fact = doc_term_matrix.shape[0]
    
    dict_freq = dict()
    for w in vectorizer.vocabulary_:
        dict_freq[w] = ngram_freqs[vectorizer.vocabulary_[w]]/norm_fact
        
    normalized_freq = sorted(dict_freq.items(), key=lambda kv: kv[1], reverse=True)
    
    return normalized_freq

def frequency_plot(words_noncritical, words_critical):
    noncritical_wf = compute_word_freq(words_noncritical)
    critical_wf = compute_word_freq(words_critical)
    
    noncritical_wf_dict = dict(noncritical_wf)
    for sana, _ in critical_wf:

        if sana not in noncritical_wf_dict:
            noncritical_wf_dict[sana] = 0

    noncritical_wf_plot = [(a, noncritical_wf_dict[a]) for a, _ in critical_wf]
    
    fig = plt.figure(figsize = [15,4]) 
    plt.bar(range(len(critical_wf[:25])), [val[1] for val in critical_wf[:25]], align='center', alpha=1, color="darkred")
    plt.bar(range(len(noncritical_wf_plot[:25])), [val[1] for val in noncritical_wf_plot[:25]], align='center', alpha=0.6, color="darkgreen")
    plt.xticks(range(len(critical_wf[:25])), [val[0] for val in critical_wf[:25]])
    plt.xticks(rotation=70)
    plt.show()
    
def get_vectorization(corpus, min_df=5, max_ngram = 2, freq_norm=False):
    
    if freq_norm:
        vectorizer = TfidfVectorizer(min_df=min_df, lowercase=False, ngram_range=(1, max_ngram))
    else:
        vectorizer = CountVectorizer(min_df=min_df, lowercase=False, ngram_range=(1, max_ngram))
        
    doc_term_matrix = vectorizer.fit_transform(corpus) 
    
    return vectorizer, doc_term_matrix

def convert_date(x):
    return datetime.strptime(x[:-1], '%d.%m.%Y %H:%M')

def light_filter(sentence):
    pattern = r'\{.*?\}|\[.*?\]|\n|\xa0'
    pre_filt = re.sub(pattern, '', sentence)
    return pre_filt

def save_model(db, vc, dmat):
    TC_model = {"model_database": db, "model_vectorizer": vc, "model_docterm": dmat}
    pickle.dump(TC_model, open("model/TC_model.p", "wb"))
    print("Model correctly saved.")

def get_recommendations(encoded_ticket, n, doc_term_matrix, database):
    
    cos_sim = cosine_similarity(encoded_ticket, doc_term_matrix)[0]
    ind = np.argpartition(cos_sim, -4)[-4:]
    ind_sorted = np.flip(ind[np.argsort(cos_sim[ind])])
    
    similar_tickets = []
    similar_tickets_scores = []

    for sim_ticket in range(0, n):
        #print(dataset_fin[ind_sorted[sim_ticket]]["Issue key"])
        similar_tickets += [database[ind_sorted[sim_ticket]]["Issue key"]]
        #print("SCORE: {:.2f}".format(cos_sim[ind_sorted[sim_ticket]]))
        similar_tickets_scores += [round(cos_sim[ind_sorted[sim_ticket]], 2)]

    return similar_tickets, similar_tickets_scores

def get_scored_url(st, sc, prefix):

    modified_url_chunk = []

    for esd_ticket in st:
        modified_url_chunk += [str(esd_ticket) + "%2C%20"]

    remove_end = "".join(modified_url_chunk)[:-6]
    add_end = remove_end + ")"

    final_url = prefix + add_end
    max_score = max(sc)
    min_score = min(sc)

    return final_url, min_score, max_score

def get_inference_extension(dataset, prefix, n, vectorizer, doc_term_matrix, database): 

    encoded_column = dataset.apply(lambda x: vectorizer.transform([x.ProcessedSentence]), axis=1)
    inference_data = []

    for enc in encoded_column:

        similar_tickets, scores = get_recommendations(enc, n, doc_term_matrix, database)
        URL, MIN_SCORE, MAX_SCORE = get_scored_url(similar_tickets, scores, prefix)
        inference_data += [[MIN_SCORE, MAX_SCORE, URL]]

    inference_array = np.asanyarray(inference_data)
    assert inference_array.shape[0] == len(encoded_column), "TC: Dimensions do not match"
    inference_extension = pd.DataFrame({'Minimum': inference_array[:, 0], 'Maximum': inference_array[:, 1], 'Recommendations-URL': inference_array[:, 2]})

    return inference_extension