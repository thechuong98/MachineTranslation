import json
from datasets import load_dataset, load_metric
from src.datamodules.tokenizer import
from nltk import word_tokenize
from tqdm import tqdm

def load_data():
    raw_datasets = load_dataset("mt_eng_vietnamese", "iwslt2015-en-vi")

    train_src_str = [i['en'] for i in raw_datasets['train']['translation'] if
                     (len(i['en'].strip()) > 0 and len(i['vi'].strip()) > 0)]
    train_tgt_str = [i['vi'] for i in raw_datasets['train']['translation'] if
                     (len(i['en'].strip()) > 0 and len(i['vi'].strip()) > 0)]
    val_src_str = [i['en'] for i in raw_datasets['validation']['translation'] if
                   (len(i['en'].strip()) > 0 and len(i['vi'].strip()) > 0)]
    val_tgt_str = [i['vi'] for i in raw_datasets['validation']['translation'] if
                   (len(i['en'].strip()) > 0 and len(i['vi'].strip()) > 0)]
    test_src_str = [i['en'] for i in raw_datasets['test']['translation'] if
                    (len(i['en'].strip()) > 0 and len(i['vi'].strip()) > 0)]
    test_tgt_str = [i['vi'] for i in raw_datasets['test']['translation'] if
                    (len(i['en'].strip()) > 0 and len(i['vi'].strip()) > 0)]

    return {'train': (train_src_str, train_tgt_str),
            'val': (val_src_str, val_tgt_str),
            'test': (test_src_str, test_tgt_str)}

def process_store_data():

