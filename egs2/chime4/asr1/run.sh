#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="tr05_simu_isolated_1ch_track"
valid_set="dt05_simu_isolated_1ch_track"
test_sets="et05_simu_isolated_1ch_track"

# train_set=tr05_multi_noisy_si284 # tr05_multi_noisy (original training data) or tr05_multi_noisy_si284 (add si284 data)
# valid_set=dt05_multi_isolated_1ch_track
# test_sets="\
# dt05_real_isolated_1ch_track dt05_simu_isolated_1ch_track et05_real_isolated_1ch_track et05_simu_isolated_1ch_track \
# dt05_real_beamformit_2mics dt05_simu_beamformit_2mics et05_real_beamformit_2mics et05_simu_beamformit_2mics \
# dt05_real_beamformit_5mics dt05_simu_beamformit_5mics et05_real_beamformit_5mics et05_simu_beamformit_5mics \
# "

asr_config=conf/train_asr_transformer.yaml
inference_config=conf/decode_asr_transformer.yaml
lm_config=conf/train_lm.yaml


use_word_lm=false
word_vocab_size=65000

./asr.sh                                   \
    --ngpu 2 \
    --num_nodes 2 \
    --lang en \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --nlsyms_txt data/nlsyms.txt           \
    --token_type phn                      \
    --g2p g2p_en_no_space \
    --feats_type raw               \
    --asr_config "${asr_config}"           \
    --inference_config "${inference_config}"     \
    --lm_config "${lm_config}"             \
    --use_word_lm ${use_word_lm}           \
    --word_vocab_size ${word_vocab_size}   \
    --train_set "${train_set}"             \
    --valid_set "${valid_set}"             \
    --test_sets "${test_sets}"             \
    --lm_train_text "${train_set}/text" \
    "$@"
