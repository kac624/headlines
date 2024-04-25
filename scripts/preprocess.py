# core
import pandas as pd
import json
import torch
import time
import os

# file download
from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

# sklearn
from sklearn.model_selection import train_test_split

# transformers
import transformers
from transformers import AutoTokenizer

# custom
from utils import augment_dataset, random_delete, random_replace, back_translate
from utils import consolidated_categories, performant_classes

# warnings
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message=".*pip install sacremoses.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*mkl-service package failed to import.*")
transformers.logging.set_verbosity_error()



"""SETUP"""

start = time.time()
os.makedirs('data/processed', exist_ok=True)



"""CONFIG"""

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

with open('scripts/_config.json', 'r') as file:
    config = json.load(file)

CONSOLIDATE_LABELS = config['CONSOLIDATE_LABELS']
CONSOLIDATE_OTHER = config['CONSOLIDATE_OTHER']
DATA_AUGMENTATION = config['DATA_AUGMENTATION']
TRAIN_SPLIT_PROP = config['TRAIN_SPLIT_PROP']
PRETRAINED_LM = config['PRETRAINED_LM']
MAX_LENGTH = config['MAX_LENGTH']

RANDOM_SEED = config['RANDOM_SEED']
torch.manual_seed(RANDOM_SEED)



"""LOAD DATA, CONSOLIDATE AND SPLIT"""

# download dataset with Kaggle API
api.dataset_download_file('rmisra/news-category-dataset/','News_Category_Dataset_v3.json')

# designate downloaded file as zip, and unzip
zf = ZipFile('News_Category_Dataset_v3.json.zip')
zf.extractall()
zf.close()

# read in data as df
news = pd.read_json('News_Category_Dataset_v3.json', orient='records', lines=True)
print(f'Loaded {len(news)} records with {news.category.nunique()} categories')

# delete downloaded zip and extracted csv - keep your directory clean!
os.remove('News_Category_Dataset_v3.json.zip')
os.remove('News_Category_Dataset_v3.json')

# Mark performant classes
news['performant'] = news.category.isin(performant_classes)

# Consolidate categories with custom classes
if CONSOLIDATE_LABELS:
    # Invert the dictionary to map each old label to its new, consolidated label
    category_mapping = {
        old_label: new_label for new_label, old_labels in consolidated_categories.items() for old_label in old_labels
    }
    # Apply the mapping to the 'news' DataFrame as before
    news.category = news.category.map(category_mapping)
    # Consolidate non-performant classes under Other
    if CONSOLIDATE_OTHER:
        news.loc[~news.performant, 'category'] = 'OTHER'

    print(f'Consolidated labels to {len(news.category.unique())} classes')

# Collect label names and conversion dict
label_names = news.category.unique()
code_labels_dict = dict(enumerate(news.category.astype('category').cat.categories))

# Integer encode labels and combine two text columns
news['label'] = news.category.astype('category').cat.codes
news['text'] = news.headline + ' ' + news.short_description

# Split
train, valid = train_test_split(news, test_size=1 - TRAIN_SPLIT_PROP, stratify=news.category, random_state=RANDOM_SEED)
valid, test = train_test_split(valid, test_size=0.5, stratify=valid.category, random_state=RANDOM_SEED)

print(f'Finished Split - Train: {len(train)}, Test: {len(test)}, Valid: {len(valid)}')



"""DATA AUGMENTATION"""

# Check config for data augmentations
if DATA_AUGMENTATION:
    # Set up variables
    frac = DATA_AUGMENTATION['PERCENTAGE']
    rows_added = 0
    if DATA_AUGMENTATION['BACKTRANSLATION']:
        train, translate_rows_added = augment_dataset(train, back_translate, frac, device=device, MAX_LENGTH=MAX_LENGTH)
        rows_added += translate_rows_added
    if DATA_AUGMENTATION['DELETION']:
        train, translate_rows_added = augment_dataset(train, random_delete, frac)
        rows_added += translate_rows_added
    if DATA_AUGMENTATION['REPLACEMENT']:
        train, translate_rows_added = augment_dataset(train, random_replace, frac)
        rows_added += translate_rows_added
    print(
        f'Augmented {rows_added} samples - Time Elapsed - {(time.time()-start) / 60:.2f} minutes\n'
        f'Split with Augments - Train: {len(train)}, Test: {len(test)}, Valid: {len(valid)}'
    )



"""LABELS"""

y_train = torch.LongTensor(train['label'].values.tolist())
y_valid = torch.LongTensor(valid['label'].values.tolist())
y_test = torch.LongTensor(test['label'].values.tolist())

print(f'Finished Labels - Train: {len(y_train)}, Test: {len(y_test)}, Valid: {len(y_valid)}')



"""TF-IDF"""

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(min_df=3, stop_words='english', max_features=5000, ngram_range=(1, 3))

tfidf_train = tfidf.fit_transform(train.text)
tfidf_train = torch.from_numpy(tfidf_train.toarray()).to(dtype=torch.float32)

tfidf_valid = tfidf.transform(valid.text)
tfidf_valid = torch.from_numpy(tfidf_valid.toarray()).to(dtype=torch.float32)

tfidf_test = tfidf.transform(test.text)
tfidf_test = torch.from_numpy(tfidf_test.toarray()).to(dtype=torch.float32)

print(f'Finished TF-IDF - Train: {tfidf_train.shape}, Test: {tfidf_test.shape}, Valid: {tfidf_valid.shape}')



"""LLM - TOKENIZE"""

transformers.logging.set_verbosity_error()
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_LM)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = 'right'
if 'bert' in PRETRAINED_LM.lower():
    tokenizer.do_lower_case = True

train_inputs = tokenizer(train['text'].values.tolist(), return_tensors='pt', padding='max_length', max_length=MAX_LENGTH, truncation=True)
valid_inputs = tokenizer(valid['text'].values.tolist(), return_tensors='pt', padding='max_length', max_length=MAX_LENGTH, truncation=True)
test_inputs = tokenizer(test['text'].values.tolist(), return_tensors='pt', padding='max_length', max_length=MAX_LENGTH, truncation=True)

train_input_ids = train_inputs['input_ids']
train_att_masks = train_inputs['attention_mask']

valid_input_ids = valid_inputs['input_ids']
valid_att_masks = valid_inputs['attention_mask']

test_input_ids = test_inputs['input_ids']
test_att_masks = test_inputs['attention_mask']

print(f'Finished LLM Data - Train: {train_input_ids.shape}, Test: {valid_input_ids.shape}, Valid: {test_input_ids.shape}')



"""SAVE"""

with open('data/code_labels_dict.json', 'w') as f:
    json.dump(code_labels_dict, f)

torch.save(y_train, 'data/y_train.pt')
torch.save(y_valid, 'data/y_valid.pt')
torch.save(y_test, 'data/y_test.pt')

torch.save(tfidf_train, 'data/tfidf_train.pt')
torch.save(tfidf_valid, 'data/tfidf_valid.pt')
torch.save(tfidf_test, 'data/tfidf_test.pt')

torch.save(train_input_ids, 'data/train_input_ids.pt')
torch.save(train_att_masks, 'data/train_att_masks.pt')

torch.save(valid_input_ids, 'data/valid_input_ids.pt')
torch.save(valid_att_masks, 'data/valid_att_masks.pt')

torch.save(test_input_ids, 'data/test_input_ids.pt')
torch.save(test_att_masks, 'data/test_att_masks.pt')


"""END"""

print(f'Preprocessing Script Complete - Time Elapsed - {(time.time()-start) / 60:.2f} minutes\n')