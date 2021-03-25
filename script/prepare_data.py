import json
from datasets import load_dataset, load_metric
from src.datamodules.tokenizer import
from nltk import word_tokenize
from tqdm import tqdm
from tokenizers.processors import TemplateProcessing
from tokenizers import ByteLevelBPETokenizer




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

