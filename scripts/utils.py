import pandas as pd 
import numpy as np
from itertools import chain

# import transformers
# transformers.logging.set_verbosity_error()
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet


"""PREPROCESSING FUNCTIONS"""

def augment_dataset(train, function, frac, **kwargs):
    # sample rows from non-performant classes
    rows_to_augment = train[train.performant == False].groupby('category').apply(lambda x: x.sample(frac=frac))
    # get text from those rows
    text_to_alter = rows_to_augment.text.to_list()
    # alter text
    altered_text = function(text_to_alter, **kwargs)
    # add altered back text to dataframe
    rows_to_augment.text = altered_text
    train = pd.concat([train, rows_to_augment], ignore_index=True)
    # count number of rows
    rows_added = len(rows_to_augment)

    return train, rows_added


def back_translate(texts, device, MAX_LENGTH, batch_size=64):
    # Load models
    to_target = pipeline(
        'translation_en_to_fr', model='Helsinki-NLP/opus-mt-en-fr', 
        device=device, batch_size=batch_size, framework='pt'
    )  
    to_source = pipeline(
        'translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en',
        device=device, batch_size=batch_size, framework='pt'
    )
    # Translate
    translated_to_target = to_target(texts, max_length=MAX_LENGTH)
    translated_to_target = [x['translation_text'] for x in translated_to_target]
    # Back-translate
    back_translated_texts = to_source(translated_to_target, max_length=MAX_LENGTH)
    back_translated_texts = [x['translation_text'] for x in back_translated_texts]
    
    return back_translated_texts


def random_delete(texts, p=0.2):
    altered_texts = []
    # Loop through texts
    for text in texts:
        # Tokenize then rejoin. Delete with probability p
        altered_text = ' '.join([x if np.random.random() > p else '' for x in text.split()])
        # If too short, keep original
        if len(altered_text) < 5:
            altered_text = text
        # Append to list
        altered_texts.append(altered_text)
    
    return altered_texts


def random_replace(texts, p=0.2):
    altered_texts = []
    for text in texts:
        altered_text = ''
        for word in text.split():
            if np.random.random() < p:
                synonyms = set(chain.from_iterable([syn.lemma_names() for syn in wordnet.synsets(word)]))
                word = np.random.choice(list(synonyms)) if synonyms else word
            altered_text += ' ' + word
        # Append to list
        altered_texts.append(altered_text)
    
    return altered_texts


def get_bert_inputs(tokenizer, docs, MAX_LENGTH=128):
    encoded_dict = tokenizer.batch_encode_plus(
        docs, add_special_tokens=True, max_length=MAX_LENGTH, padding='max_length',
        return_attention_mask=True, truncation=True, return_tensors='pt'
    )
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    
    return input_ids, attention_masks



"""CATEGORIES"""

consolidated_categories = {
    'Diversity': ['BLACK VOICES', 'LATINO VOICES', 'QUEER VOICES'],
    'Politics': ['POLITICS'],
    'Divorce': ['DIVORCE'],
    'Weddings': ['WEDDINGS'],
    'Home & Living': ['HOME & LIVING'],
    'Travel': ['TRAVEL'],	
    'Style & Beauty': ['STYLE & BEAUTY', 'STYLE'],
    'Food & Drink': ['FOOD & DRINK', 'TASTE'],
    'Sports': ['SPORTS'],
    'Entertainment, Art and Media': ['ENTERTAINMENT', 'MEDIA', 'COMEDY', 'ARTS', 'ARTS & CULTURE', 'CULTURE & ARTS'],
    'Wellness': ['WELLNESS'],
    'General News': ['WORLD NEWS', 'U.S. NEWS', 'WORLDPOST', 'THE WORLDPOST'],
    'Parenting': ['PARENTING', 'PARENTS'],
    'Business': ['BUSINESS', 'MONEY', 'TECH'],
    ##### NON-PERFORMANT CLASSES
    'Other': ['IMPACT',  'GOOD NEWS', 'WEIRD NEWS'],
    'Healthy Living': ['FIFTY', 'HEALTHY LIVING'],
    'Education': ['EDUCATION', 'COLLEGE'],
    'Crime': ['CRIME'],
    'Environment and Science': ['ENVIRONMENT', 'GREEN', 'SCIENCE'],
    'Religion': ['RELIGION'],
    'Women': ['WOMEN']
}

performant_classes = [
    'BLACK VOICES', 'LATINO VOICES', 'QUEER VOICES', 'POLITICS',
    'DIVORCE', 'WEDDINGS', 'HOME & LIVING', 'TRAVEL', 'STYLE & BEAUTY', 'STYLE',
    'FOOD & DRINK', 'TASTE', 'SPORTS', 'ENTERTAINMENT', 'MEDIA', 'COMEDY',
    'WELLNESS', 'WORLD NEWS', 'U.S. NEWS', 'WORLDPOST', 'THE WORLDPOST',
    'PARENTING', 'PARENTS', 'ARTS', 'ARTS & CULTURE', 'CULTURE & ARTS'
]