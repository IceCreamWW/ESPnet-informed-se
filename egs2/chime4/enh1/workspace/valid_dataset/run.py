from espnet2.text.phoneme_tokenizer import PhonemeTokenizer
from utils import *
import argparse
import pdb



tokenizer = PhonemeTokenizer(g2p_type="g2p_en_no_space")
def valid_data(uttid, text, ali):
    informed_words = text.strip().split()
    informed_ali_ref = ali.strip().split()
    percent = 100

    informed_ali_ctm = ali2ctm(informed_ali_ref, sil='SIL')

    num_words = int(min(max(len(informed_words) * percent, 1), len(informed_words)))
    start = random.randint(0, len(informed_words) - num_words)
    partial_text = " ".join(informed_words[start:start + num_words])
    partial_phns = tokenizer.text2tokens(partial_text)

    informed_ali_seq = [t[0] for t in informed_ali_ctm]
    occurences = find_occurences(informed_ali_seq, partial_phns, informed_ali_ctm)

    if occurences == 0:
        print(uttid)
    # assert len(occurences) > 0, uttid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='validate dataset for partially informed training')
    parser.add_argument('--text_raw', type=str)
    parser.add_argument('--text_phn_ali', type=str)
    args = parser.parse_args()


    with open(args.text_raw) as fp:
        text_raw = fp.read().strip().split('\n')

    with open(args.text_phn_ali) as fp:
        text_phn_ali = fp.read().strip().split('\n')


    for i, (raw, phn_ali) in enumerate(zip(text_raw, text_phn_ali), 1):
        if i % 100 == 0:
            print(i)
        uttid, raw = raw.strip().split(maxsplit=1)
        phn_ali = phn_ali.strip().split(maxsplit=1)[1]
        valid_data(uttid, raw, phn_ali)



