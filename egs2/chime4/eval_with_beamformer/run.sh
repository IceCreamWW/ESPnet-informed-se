python=python3
org_set=dt05_simu_isolated_1ch_track
enh_set=dt05_simu_isolated_1ch_track_enh_ae_student
expdir=exp/beamforming_${enh_set}/


outdir=${expdir}/beamformed/


${python} -m espnet2.bin.eval_with_beamformer \
    --train_config /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet_recipe/egs2/conferencingspeech2021/enh1/exp/enh_mpdr_beamformer/config.yaml \
    --do_pre_masking true \
    --outdir ${outdir} \
    --wavscp data/${org_set}/wav.scp \
    --spkscp data/${org_set}/spk1.scp \
    --enhscp data/${enh_set}/wav.scp \
    --write_wavs true

