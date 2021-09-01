#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=16000
n_fft=2048
n_shift=300
win_length=1200

opts=
if [ "${fs}" -eq 16000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_set=tr05_simu_isolated_1ch_track_ref
valid_set=dt05_simu_isolated_1ch_track_ref
test_sets="et05_simu_isolated_1ch_track_ref"

train_config=conf/train.yaml
inference_config=conf/decode.yaml

cleaner=tacotron
g2p=g2p_en_no_space # or g2p_en
local_data_opts="--trim_all_silence true" # trim all silence in the audio

./tts.sh \
    --ngpu 4 \
    --lang en \
    --feats_type raw \
    --local_data_opts "${local_data_opts}" \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --use_xvector true \
    --token_type phn \
    --cleaner "${cleaner}" \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    ${opts} "$@"
