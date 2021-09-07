from TC_helper_functions import * 

DATA_RAW_DIR = "data/01_raw"

all_files = [filename for filename in os.listdir(DATA_RAW_DIR) if filename.endswith(".csv")]
df_original = pd.concat((pd.read_csv(os.path.join(DATA_RAW_DIR, f)) for f in all_files))

print("{} files have been loaded.".format(len(all_files)))
print("Number of tickets before any filtering: {}.".format(len(df_original)))

# RESOLVED & NON-EVENT TICKETS
df = df_original[df_original['Status'] == "Resolved"].copy()
print("Number of resolved tickets: {}.".format(len(df)))
df = df[df['Labels'] != "event"]
print("Number of non-event tickets: {}.".format(len(df)))

# SELECT ISSUE TYPE
ISSUE_TYPE = "Service Request"

print("Possible types of issues: {}.".format(set(df["Issue Type"])))
df = df[df["Issue Type"] == ISSUE_TYPE]
print("\nNumber of {} type of issues: {}.".format(ISSUE_TYPE, len(df)))

# REARRANGE
df = df[['Reporter','Assignee', "Custom field (Customer)", "Summary", "Description", "Created", "Resolved", "Issue key"]]

# DROP CRITC. NAN
df = df.dropna(subset=['Summary', 'Description', 'Created', 'Resolved'])
print("Number of tickets after removing samples with nan: {}.".format(len(df)))

# LABEL CRITICALITY
SPAN_THRESHOLD = 7
created_do = df['Created'].apply(convert_date)
resolved_do = df["Resolved"].apply(convert_date)
df["Span"] = [t.days for t in resolved_do-created_do]
df['Critical'] = (df['Span'] > SPAN_THRESHOLD).astype(int)
print("Number of tickets {}, of which {} ({:.2f}) are labeled critical.".format(len(df), len(df[df["Critical"]==1]), len(df[df["Critical"]==1])/len(df)))
df = df.reset_index(drop=True)

# OPTIONAL: Select Customer
#CUSTOMER = "Efima (000)"
#df_EFIMA = df[df["Custom field (Customer)"] == CUSTOMER].copy()
#print("Number of tickets {}, of which {} ({:.2f}) are labeled critical for {}.".format(len(df_EFIMA), len(df_EFIMA[df_EFIMA["Critical"]==1]), len(df_EFIMA[df_EFIMA["Critical"]==1])/len(df_EFIMA), CUSTOMER))
#df_EFIMA = df_EFIMA.reset_index(drop=True)

# DEFINE & LOAD STOPWORDS
DATA_STOPWORDS_DIR = "data/02_auxiliary"

stopwords_set = load_stopwords(DATA_STOPWORDS_DIR, general=True, it=True, names=True)
print("Number of stopwords loaded: {}.".format(len(stopwords_set)))

# RUN NLP-process
dataset = df.copy()
#dataset = df_EFIMA.copy()

dataset["ProcessedWords"] = dataset.apply(lambda x: remove_stopwords(get_nlp_processed_text(x.Summary, x.Description), stopwords_set), axis=1)
dataset["ProcessedSentence"] =  dataset.apply(lambda x: ' '.join(x.ProcessedWords), axis=1)
#dataset['Language'] = dataset.apply(lambda x: detect_language(x.ProcessedSentence), axis=1)

# KEEP FINNISH ONLY
#dataset_fin = dataset[dataset["Language"] == "fi"].reset_index(drop=True)
#print("Number of Finnish tickets: {}.".format(len(dataset_fin)))

# CONVERT TO BoW TYPE
corpus = dataset.ProcessedSentence
vectorizer, doc_term_mat = get_vectorization(corpus, min_df=3, max_ngram=2, freq_norm=False)

# SAVE MODEL
main_data_source = dataset.to_dict(orient="index")
save_model(main_data_source, vectorizer, doc_term_mat)