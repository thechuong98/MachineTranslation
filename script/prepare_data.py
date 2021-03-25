import json
from datasets import load_dataset, load_metric
from nltk import word_tokenize
from tqdm import tqdm
from tokenizers.processors import TemplateProcessing
from tokenizers import ByteLevelBPETokenizer
import torch
import os

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

def training_tokenizers(raw_datasets: dict, hparams: dict):
    train_dataset, val_dataset, test_dataset = raw_datasets[
        'train'], raw_datasets['val'], raw_datasets['test']
    en_tokenizer = ByteLevelBPETokenizer()
    en_tokenizer.train_from_iterator(
        train_dataset[0] + val_dataset[0],
        vocab_size=hparams['src_vocab_size'],
        min_frequency=2,
        special_tokens=["[BOS]", "[PAD]", "[EOS]", "[UNK]"]
    )
    en_tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", en_tokenizer.token_to_id("[BOS]")),
            ("[EOS]", en_tokenizer.token_to_id("[EOS]")),
        ],
    )
    en_tokenizer.enable_truncation(max_length=hparams['seq_len'])
    en_tokenizer.enable_padding(
        direction='right', length=hparams['seq_len'], pad_id=en_tokenizer.get_vocab()["[PAD]"]
    )
    en_tokenizer.save_model("src/tokenizers/en")

    vi_tokenizer = ByteLevelBPETokenizer()
    vi_tokenizer.train_from_iterator(
        train_dataset[1] + val_dataset[1],
        vocab_size=hparams['tgt_vocab_size'],
        min_frequency=2,
        special_tokens=["[BOS]", "[PAD]", "[EOS]", "[UNK]"],
    )
    vi_tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", vi_tokenizer.token_to_id("[BOS]")),
            ("[EOS]", vi_tokenizer.token_to_id("[EOS]"))
        ]
    )
    vi_tokenizer.enable_truncation(max_length=hparams['seq_len']+1)
    vi_tokenizer.enable_padding(
        direction='right', length=hparams['seq_len']+1, pad_id=vi_tokenizer.get_vocab()["[PAD]"]
    )
    vi_tokenizer.save_model("src/tokenizer/vi")

    return en_tokenizer, vi_tokenizer

def dump_json(strs, ids, outpath):
    output = []
    for str, id in zip(strs, ids):
        sample = {
            'text': str,
            'token_ids': id
        }
        output.append(sample)
    json.dump(output, open(outpath, 'w'), ensure_ascii=False)

def tokenize_data(raw_datasets: dict, en_tokenizer, vi_tokenizer, outpath: str):
    train_src_str = raw_datasets[0][0]
    train_tgt_str = raw_datasets[0][1]
    val_src_str = raw_datasets[1][0]
    val_tgt_str = raw_datasets[1][1]
    test_src_str = raw_datasets[2][0]
    test_tgt_str = raw_datasets[2][1]

    train_src_ids = [i.ids for i in en_tokenizer.encode_batch(train_src_str)]
    train_tgt_ids = [i.ids for i in vi_tokenizer.encode_batch(train_tgt_str)]
    val_src_ids = [i.ids for i in en_tokenizer.encode_batch(val_src_str)]
    val_tgt_ids = [i.ids for i in vi_tokenizer.encode_batch(val_tgt_str)]
    test_src_ids = [i.ids for i in en_tokenizer.encode_batch(test_src_str)]
    test_tgt_ids = [i.ids for i in vi_tokenizer.encode_batch(test_tgt_str)]

    dump_json(train_src_str, train_src_ids, os.path.join(outpath, 'train_src.json'))
    dump_json(train_tgt_str, train_tgt_ids, os.path.join(outpath, 'train_tgt.json'))
    dump_json(val_src_str, val_src_ids, os.path.join(outpath, 'val_src.json'))
    dump_json(val_tgt_str, val_tgt_ids, os.path.join(outpath, 'val_tgt.json'))
    dump_json(test_src_str, test_src_ids, os.path.join(outpath, 'test_src.json'))
    dump_json(test_tgt_str, test_tgt_ids, os.path.join(outpath, 'test_tgt.json'))

if __name__ == '__main__':
    outp = 'data/processed_iwslt_data/'
    raw_datasets = load_data()
    en_tokenizer, vi_tokenizer = training_tokenizers(raw_datasets, hparams)
    tokenize_data(raw_datasets, en_tokenizer, vi_tokenizer, outp)
