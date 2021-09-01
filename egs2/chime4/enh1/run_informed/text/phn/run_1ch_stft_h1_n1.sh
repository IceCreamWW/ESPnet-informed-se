#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min_or_max=min # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.
sample_rate=16k


train_set="tr05_simu_isolated_1ch_track"
valid_set="dt05_simu_isolated_1ch_track"
# test_sets="tr05_real_isolated_1ch_track"
# test_sets="et05_simu_isolated_1ch_track tr05_multi_noisy_si284"
test_sets="et05_simu_isolated_1ch_track"

./enh_ti.sh \
    --ngpu 1 \
    --num_nodes 1 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --enh_tag "stft_tasnet_informed_text_phn_1ch_h1_n1" \
    --channels 1 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --fs ${sample_rate} \
    --audio_format wav \
    --feats_type raw \
    --spk_num 1 \
    --enh_config ./conf/tuning/train_enh_stft_tasnet_phn_informed_h1_n1.yaml \
    --token_list data/token_list/phn/tokens.txt \
    --token_type phn \
    --g2p g2p_en_no_space \
    --lm_train_text data/${train_set}/text \
    --inference_model "valid.loss.best.pth" \
    "$@"
