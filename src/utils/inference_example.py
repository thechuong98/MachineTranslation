from PIL import Image

from src.models.mnist_model import LitModelMNIST
from src.models.nmt_model import NMTLitModel
from src.transforms import mnist_transforms
import torch
import pdb
import numpy as np

def predict():
    """Example of inference with trained model.
    It loads trained image classification model from checkpoint.
    Then it loads example image and predicts its label.
    """

    # ckpt can be also a URL!
    CKPT_PATH = "last.ckpt"

    # load model from checkpoint
    # model __init__ parameters will be loaded from ckpt automatically
    # you can also pass some parameter explicitly to override it
    trained_model = LitModelMNIST.load_from_checkpoint(checkpoint_path=CKPT_PATH)

    # print model hyperparameters
    print(trained_model.hparams)

    # switch to evaluation mode
    trained_model.eval()
    trained_model.freeze()

    # load data
    img = Image.open("data/example_img.png").convert("L")  # convert to black and white
    # img = Image.open("data/example_img.png").convert("RGB")  # convert to RGB

    # preprocess
    img = mnist_transforms.mnist_test_transforms(img)
    img = img.reshape((1, *img.size()))  # reshape to form batch of size 1

    # inference
    output = trained_model(img)
    print(output)


CKPT_PATH = "../../logs/runs/2021-04-01/12-12-03/checkpoints/last.ckpt"
litmodel = NMTLitModel.load_from_checkpoint(checkpoint_path=CKPT_PATH)


def translate(sentences, lightning_module):
    src_ids = [i.ids for i in lightning_module.src_tokenizer.encode_batch(sentences)]
    src_ids = torch.LongTensor(src_ids).to(lightning_module.device) # (B, seq_len)
    src_mask = lightning_module.model.make_src_mask(src_ids)

    encoder_output = lightning_module.model.encode(src_ids, src_mask)
    tgt_init_str = [""] * len(sentences)
    # tgt_tokenizer will return seq_len + 1 tokens, so we strip one here
    tgt_ids = [i.ids[:-1] for i in lightning_module.tgt_tokenizer.encode_batch(tgt_init_str)]
    tgt_ids = np.array(tgt_ids)
    tgt_eos_idx = lightning_module.tgt_tokenizer.get_vocab()["[EOS]"]

    tgt_mask = lightning_module.model.casual_mask.unsqueeze(0).to(lightning_module.device) # (1, seq_len, seq_len)
    completed = np.array([False for _ in sentences])

    for step_idx in range(lightning_module.hparams.seq_len - 1):
        tgt_ids_inp = torch.LongTensor(tgt_ids).to(lightning_module.device)
        logits, attentions = lightning_module.model.decode(encoder_output, src_mask, tgt_ids_inp, tgt_mask)
        # logits: B, seq_len, vocab_size
        current_decode_step_logits = logits[:, step_idx] # (B, V)
        _, best_ids = current_decode_step_logits.topk(1, dim=-1) # argmax on dim vocab
        best_ids = best_ids.detach().cpu().numpy().flatten() # (V,)
        tgt_ids[:, step_idx+1] = best_ids # Set best tokens (output of current step) to input of next step
        completed = completed | (best_ids == tgt_eos_idx) # update `completed` when best_ids[i] == EOS
        if all(completed):
            break

    # Strip [EOS] token for all tgt_ids
    predict_sentences = []
    strip_tgt_ids = []
    for tgt_sentence in tgt_ids:
        # Strip all token after EOS
        strip_ids = tgt_sentence
        if tgt_eos_idx in tgt_sentence:
            eos_idx = tgt_sentence.tolist().index(tgt_eos_idx)
            strip_ids = tgt_sentence[:eos_idx + 1] # Include <EOS>
        strip_tgt_ids.append(strip_ids.tolist())
        predict_sentences.append( lightning_module.tgt_tokenizer.decode(strip_ids) )
    return strip_tgt_ids, predict_sentences, attentions


if __name__ == "__main__":
    src_sentence = [
        "I'm a little busy right now.",
    ]

    tgt_ids, predict_sentences, attentions = translate(src_sentence, litmodel)
    for src, pred in zip(src_sentence, predict_sentences):
        print(src, "--->", pred)

