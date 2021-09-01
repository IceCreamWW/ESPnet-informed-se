#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=8k
# Path to a directory containing extra annotations for CHiME4
# Run `local/data.sh` for more information.

train_set=tr_mix_single_min_8k
valid_set=cv_mix_single_min_8k
test_sets="tt_mix_single_min_8k"

    # --speed_perturb_factors "0.9 1.0 1.1" \
./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 2 \
    --spk_num 1 \
    --enh_config conf/tuning/train_enh_conv_tasnet.yaml \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    "$@"
