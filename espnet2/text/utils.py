import numpy as np
import random
from espnet2.text.abs_tokenizer import AbsTokenizer
from espnet2.text.raw_tokenizer import RawTokenizer
from espnet2.text.char_tokenizer import CharTokenizer
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet2.text.word_tokenizer import WordTokenizer
import pdb

def ali2ctm(ali, sil=1):
    ctm = []
    phn = sil
    idx = 0
    for i, e in enumerate(ali):
        if e != phn:
            if phn != sil:
                ctm.append((phn, idx, i))
            phn = e
            idx = i
    ctm.append((phn, idx, i))
    return ctm

def recompute_ctm(ctm, frame_len_old, frame_shift_old, frame_len_new, frame_shift_new):
    assert frame_shift_old / frame_shift_new == frame_shift_old // frame_shift_new
    offset = round(((frame_len_old - frame_shift_old) / 2000  - (frame_len_new - frame_shift_new) / 2000) / frame_shift_new)
    repeats = frame_shift_old // frame_shift_new
    for i in range(len(ctm)):
        c, start, end = ctm[i]
        ctm[i] = (c, start * repeats + offset, end * repeats + offset)
    return ctm

def find_occurences(sequence, target, ctm = None):
    occurences = [(i, i + len(target) - 1) for i in range(len(sequence) - len(target) + 1) if sequence[i:i+len(target)] == target]
    if ctm is not None:
        for i in range(len(occurences)):
            start, end = occurences[i]
            occurences[i] = (ctm[start][1], ctm[end][2])
    return occurences

def mask_occurences_in_sequence(sequence, occurences):
    mask = np.zeros_like(sequence)
    for occurence in occurences:
        start, end = occurence
        mask[start:end] = 1
    return sequence * mask

