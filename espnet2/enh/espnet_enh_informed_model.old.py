from functools import reduce
from itertools import permutations
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
import pdb


ALL_LOSS_TYPES = (
    # mse_loss(predicted_mask, target_label)
    "mask_mse",
    # mse_loss(enhanced_magnitude_spectrum, target_magnitude_spectrum)
    "magnitude",
    # mse_loss(enhanced_complex_spectrum, target_complex_spectrum)
    "spectrum",
    # log_mse_loss(enhanced_complex_spectrum, target_complex_spectrum)
    "spectrum_log",
    # si_snr(enhanced_waveform, target_waveform)
    "si_snr",
)


class ESPnetEnhancementTIModel(AbsESPnetModel):
    """Speech enhancement or separation Frontend model"""

    def __init__(
        self,
        enh_model: Optional[AbsEnhancement],
        stft_consistency: bool = False,
        component_loss: bool = False
    ):
        assert check_argument_types()

        super().__init__()

        self.enh_model = enh_model
        self.num_spk = enh_model.num_spk
        self.num_noise_type = getattr(self.enh_model, "num_noise_type", 1)
        self.component_loss = component_loss

        # get mask type for TF-domain models (only used when loss_type="mask_*")
        self.mask_type = getattr(self.enh_model, "mask_type", None)
        # get loss type for model training
        self.loss_type = getattr(self.enh_model, "loss_type", None)
        # whether to compute the TF-domain loss while enforcing STFT consistency

        if stft_consistency and loss_type in ["mask_mse", "si_snr"]:
            raise ValueError(
                f"stft_consistency will not work when '{loss_type}' loss is used"
            )

        assert self.loss_type in ALL_LOSS_TYPES, self.loss_type
        # for multi-channel signal
        self.ref_channel = getattr(self.separator, "ref_channel", -1)

    def load(self, *custom_pretrain_paths):
        if custom_pretrain_paths:
            self.enh_model.load(*custom_pretrain_paths)

    @staticmethod
    def _create_mask_label(mix_spec, ref_spec, mask_type="IAM"):
        """Create mask label.

        Args:
            mix_spec: ComplexTensor(B, T, F)
            ref_spec: List[ComplexTensor(B, T, F), ...]
            mask_type: str
        Returns:
            labels: List[Tensor(B, T, F), ...] or List[ComplexTensor(B, T, F), ...]
        """

        assert mask_type in [
            "IBM",
            "IRM",
            "IAM",
            "PSM",
            "NPSM",
            "PSM^2",
        ], f"mask type {mask_type} not supported"
        eps = 10e-8
        mask_label = []
        for r in ref_spec:
            mask = None
            if mask_type == "IBM":
                flags = [abs(r) >= abs(n) for n in ref_spec]
                mask = reduce(lambda x, y: x * y, flags)
                mask = mask.int()
            elif mask_type == "IRM":
                # TODO(Wangyou): need to fix this,
                #  as noise referecens are provided separately
                mask = abs(r) / (sum(([abs(n) for n in ref_spec])) + eps)
            elif mask_type == "IAM":
                mask = abs(r) / (abs(mix_spec) + eps)
                mask = mask.clamp(min=0, max=1)
            elif mask_type == "PSM" or mask_type == "NPSM":
                phase_r = r / (abs(r) + eps)
                phase_mix = mix_spec / (abs(mix_spec) + eps)
                # cos(a - b) = cos(a)*cos(b) + sin(a)*sin(b)
                cos_theta = (
                    phase_r.real * phase_mix.real + phase_r.imag * phase_mix.imag
                )
                mask = (abs(r) / (abs(mix_spec) + eps)) * cos_theta
                mask = (
                    mask.clamp(min=0, max=1)
                    if mask_type == "NPSM"
                    else mask.clamp(min=-1, max=1)
                )
            elif mask_type == "PSM^2":
                # This is for training beamforming masks
                phase_r = r / (abs(r) + eps)
                phase_mix = mix_spec / (abs(mix_spec) + eps)
                # cos(a - b) = cos(a)*cos(b) + sin(a)*sin(b)
                cos_theta = (
                    phase_r.real * phase_mix.real + phase_r.imag * phase_mix.imag
                )
                mask = (abs(r).pow(2) / (abs(mix_spec).pow(2) + eps)) * cos_theta
                mask = mask.clamp(min=-1, max=1)
            assert mask is not None, f"mask type {mask_type} not supported"
            mask_label.append(mask)
        return mask_label

    def forward(
        self,
        speech_mix: torch.Tensor,
        text_ref1: torch.Tensor,
        speech_mix_lengths: torch.Tensor = None,
        text_ref1_lengths: torch.Tensor = None,
        mode = -1,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            speech_mix_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
        """
        # pdb.set_trace()
        # clean speech signal of each speaker
        speech_ref = [
            kwargs["speech_ref{}".format(spk + 1)] for spk in range(self.num_spk)
        ]
        # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
        speech_ref = torch.stack(speech_ref, dim=1)

        if "noise_ref1" in kwargs:
            # noise signal (optional, required when using
            # frontend models with beamformering)
            noise_ref = [
                kwargs["noise_ref{}".format(n + 1)] for n in range(self.num_noise_type)
            ]
            # (Batch, num_noise_type, samples) or
            # (Batch, num_noise_type, samples, channels)
            noise_ref = torch.stack(noise_ref, dim=1)
        else:
            noise_ref = None

        if mode != -1:
            gan_ref = kwargs["gan_ref"]
        else:
            gan_ref = None


        # dereverberated (noisy) signal
        # (optional, only used for frontend models with WPE)
        if "dereverb_ref1" in kwargs:
            # noise signal (optional, required when using
            # frontend models with beamformering)
            dereverb_speech_ref = [
                kwargs["dereverb_ref{}".format(n + 1)]
                for n in range(self.num_spk)
                if "dereverb_ref{}".format(n + 1) in kwargs
            ]
            assert len(dereverb_speech_ref) in (1, self.num_spk), len(
                dereverb_speech_ref
            )
            # (Batch, N, samples) or (Batch, N, samples, channels)
            dereverb_speech_ref = torch.stack(dereverb_speech_ref, dim=1)
        else:
            dereverb_speech_ref = None

        batch_size = speech_mix.shape[0]
        speech_lengths = (
            speech_mix_lengths
            if speech_mix_lengths is not None
            else torch.ones(batch_size).int() * speech_mix.shape[1]
        )
        assert speech_lengths.dim() == 1, speech_lengths.shape
        # Check that batch_size is unified
        assert speech_mix.shape[0] == speech_ref.shape[0] == speech_lengths.shape[0], (
            speech_mix.shape,
            speech_ref.shape,
            speech_lengths.shape,
        )

        # for data-parallel
        speech_ref = speech_ref[:, :, : speech_lengths.max()]
        speech_mix = speech_mix[:, : speech_lengths.max()]

        # speech_mix, speech_lengths = self.ti_frontend(speech_mix, text_ref_1, speech_lengths, text_lengths)
        loss, speech_pre, mask_pre, out_lengths, perm, si_snr_loss, disc_loss, stats_ = self._compute_loss(
            speech_mix,
            text_ref1,
            speech_lengths,
            text_ref1_lengths,
            speech_ref,
            dereverb_speech_ref=dereverb_speech_ref,
            noise_ref=noise_ref,
            gan_ref=gan_ref,
            mode=mode
        )

        # pdb.set_trace()
        # add stats for logging
        if self.loss_type != "si_snr":
            if self.training:
                speech_pre = [
                    self.enh_model.stft.inverse(ps, speech_lengths)[0]
                    for ps in speech_pre
                ]
                speech_ref = torch.unbind(speech_ref, dim=1)
                if speech_ref[0].dim() == 3:
                    # For si_snr loss, only select one channel as the reference
                    speech_ref = [sr[..., self.ref_channel] for sr in speech_ref]
                # compute si-snr loss
                si_snr_loss, perm = self._permutation_loss(
                    speech_ref, speech_pre, self.si_snr_loss, perm=perm
                )
                si_snr = -si_snr_loss.detach()
                # compute si_snr(clean_mag*mix_phase, clean_mag*clean_phase) 
                # List[ComplexTensor(Batch, T, F)] or List[ComplexTensor(Batch, T, C, F)]
                spectrum_ref = [
                    ComplexTensor(*torch.unbind(self.enh_model.stft(sr)[0], dim=-1))
                    for sr in speech_ref
                ]
                spectrum_mix = self.enh_model.stft(speech_mix)[0]
                spectrum_mix = ComplexTensor(spectrum_mix[..., 0], spectrum_mix[..., 1])
                print('type of spectrum_ref: {}, type of spectrum_mix {}'.format(type(spectrum_ref),type(spectrum_mix)))
                magnitude_mix = [abs(ps) for ps in spectrum_mix]
                phase_mix = [ps/abs(ps + 1e-15) for ps in spectrum_mix]
                if spectrum_ref[0].dim() > magnitude_mix[0].dim():
                    # only select one channel as the reference
                    magnitude_ref = [
                        abs(sr[..., self.ref_channel, :]) for sr in spectrum_ref
                    ]
                else:
                    magnitude_ref = [abs(sr) for sr in spectrum_ref]
                assert len(magnitude_ref) == len(phase_mix), 'len(magnitude_ref) != len(phase_mix) {} != {}'.format(len(magnitude_ref), len(phase_mix))
                spectrum_h = [magnitude_ref[i] * phase_mix[i] for i in range(len(magnitude_ref))]
                speech_h = [
                            self.enh_model.stft.inverse(ps, speech_lengths)[0]
                            for ps in spectrum_h                   
                ]
                si_snr_loss1, perm = self._permutation_loss(
                            speech_ref, speech_h, self.si_snr_loss, perm=perm                   
                )
                si_snr1 = -si_snr_loss1.detach()
            else:
                speech_pre = [
                    self.enh_model.stft.inverse(ps, speech_lengths)[0]
                    for ps in speech_pre
                ]
                speech_ref = torch.unbind(speech_ref, dim=1)
                if speech_ref[0].dim() == 3:
                    # For si_snr loss, only select one channel as the reference
                    speech_ref = [sr[..., self.ref_channel] for sr in speech_ref]
                # compute si-snr loss
                si_snr_loss, perm = self._permutation_loss(
                    speech_ref, speech_pre, self.si_snr_loss, perm=perm
                )
                si_snr = -si_snr_loss.detach()
                si_snr1 = None

            stats = dict(
                si_snr=si_snr,
                si_snr1=si_snr1,
                loss=loss.detach(),
            )
        else:
            stats = dict(si_snr=-si_snr_loss.detach(), loss=loss.detach(), disc_loss=disc_loss.detach())
        stats.update(stats_)

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight


    def _compute_loss(
        self,
        speech_mix,
        text,
        speech_lengths,
        text_lengths,
        speech_ref,
        dereverb_speech_ref=None,
        noise_ref=None,
        gan_ref=None,
        cal_loss=True,
        mode=-1
    ):
        # mode: 0:G, 1:D, -1:normal 
        # pdb.set_trace()
        """Compute loss according to self.loss_type.

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
            speech_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            dereverb_speech_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            noise_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            cal_loss: whether to calculate enh loss, defualt is True

        Returns:
            loss: (torch.Tensor) speech enhancement loss
            speech_pre: (List[torch.Tensor] or List[ComplexTensor])
                        enhanced speech or spectrum(s)
            mask_pre: (OrderedDict) estimated masks or None
            output_lengths: (Batch,)
            perm: () best permutation
        """
        stats = dict()
        if self.component_loss:
            assert False
            assert self.num_spk == 1, self.num_spk
            speech_hat = self.enh_model.forward_rawwav(speech_ref[:, 0], text, speech_lengths, text_lengths)[0]
            loss_speech, perm_sp = self._permutation_loss(
                speech_ref.unbind(dim=1), speech_hat, self.si_snr_loss_zeromean
            )
            stats["component_speech"] = loss_speech.detach()
            if noise_ref is not None:
                noise_hat = self.enh_model.forward_rawwav(noise_ref[:, 0], text, speech_lengths, text_lengths)[0]
                loss_noise_distortion, perm_n = self._permutation_loss(
                    noise_ref.unbind(dim=1), noise_hat, self.si_snr_loss_zeromean
                )
                loss_noise_scale = (
                    torch.stack([n.pow(2).sum() for n in noise_hat], dim=0).sum()
                    / noise_hat[0].shape[0]
                )
                loss_noise = loss_noise_distortion + loss_noise_scale
                stats["component_noise"] = loss_noise.detach()
            else:
                loss_noise = 0
        else:
            loss_speech, loss_noise = 0, 0

        disc_loss = torch.tensor(0.)

        if self.loss_type != "si_snr":
            spectrum_mix = self.enh_model.stft(speech_mix)[0]
            spectrum_mix = ComplexTensor(spectrum_mix[..., 0], spectrum_mix[..., 1])

            # predict separated speech and masks
            if self.stft_consistency:
                # pseudo STFT -> time-domain -> STFT (compute loss)
                speech_pre, speech_lengths, mask_pre = self.enh_model.forward_rawwav(
                    speech_mix, speech_lengths
                )
                if speech_pre is not None:
                    spectrum_pre = []
                    for sp in speech_pre:
                        spec_pre, tf_length = self.enh_model.stft(sp, speech_lengths)
                        spectrum_pre.append(spec_pre)
                else:
                    spectrum_pre = None
                    _, tf_length = self.enh_model.stft(speech_mix, speech_lengths)
            else:
                # compute loss on pseudo STFT directly
                spectrum_pre, tf_length, mask_pre = self.enh_model(
                    speech_mix, speech_lengths
                )

            if spectrum_pre is not None and not isinstance(
                spectrum_pre[0], ComplexTensor
            ):
                spectrum_pre = [
                    ComplexTensor(*torch.unbind(sp, dim=-1)) for sp in spectrum_pre
                ]

            if not cal_loss:
                loss, perm = None, None
                return loss, spectrum_pre, mask_pre, tf_length, perm, stats

            # prepare reference speech and reference spectrum
            speech_ref = torch.unbind(speech_ref, dim=1)
            # List[ComplexTensor(Batch, T, F)] or List[ComplexTensor(Batch, T, C, F)]
            spectrum_ref = [
                ComplexTensor(*torch.unbind(self.enh_model.stft(sr)[0], dim=-1))
                for sr in speech_ref
            ]

            # compute TF masking loss
            if self.loss_type == "magnitude":
                # compute loss on magnitude spectrum
                assert spectrum_pre is not None
                magnitude_pre = [abs(ps + 1e-15) for ps in spectrum_pre]
                if spectrum_ref[0].dim() > magnitude_pre[0].dim():
                    # only select one channel as the reference
                    magnitude_ref = [
                        abs(sr[..., self.ref_channel, :]) for sr in spectrum_ref
                    ]
                else:
                    magnitude_ref = [abs(sr) for sr in spectrum_ref]

                tf_loss, perm = self._permutation_loss(
                    magnitude_ref, magnitude_pre, self.tf_mse_loss
                )
            elif self.loss_type.startswith("spectrum"):
                # compute loss on complex spectrum
                if self.loss_type == "spectrum":
                    loss_func = self.tf_mse_loss
                elif self.loss_type == "spectrum_log":
                    loss_func = self.tf_log_mse_loss
                else:
                    raise ValueError("Unsupported loss type: %s" % self.loss_type)

                assert spectrum_pre is not None
                if spectrum_ref[0].dim() > spectrum_pre[0].dim():
                    # only select one channel as the reference
                    spectrum_ref = [sr[..., self.ref_channel, :] for sr in spectrum_ref]

                tf_loss, perm = self._permutation_loss(
                    spectrum_ref, spectrum_pre, loss_func
                )
            elif self.loss_type.startswith("mask"):
                if self.loss_type == "mask_mse":
                    loss_func = self.tf_mse_loss
                else:
                    raise ValueError("Unsupported loss type: %s" % self.loss_type)

                assert mask_pre is not None
                mask_pre_ = [
                    mask_pre["spk{}".format(spk + 1)] for spk in range(self.num_spk)
                ]

                # prepare ideal masks
                mask_ref = self._create_mask_label(
                    spectrum_mix, spectrum_ref, mask_type=self.mask_type
                )

                # compute TF masking loss
                tf_loss, perm = self._permutation_loss(mask_ref, mask_pre_, loss_func)

                if "dereverb1" in mask_pre:
                    if dereverb_speech_ref is None:
                        raise ValueError(
                            "No dereverberated reference for training!\n"
                            'Please specify "--use_dereverb_ref true" in run.sh'
                        )

                    mask_wpe_pre = [
                        mask_pre["dereverb{}".format(spk + 1)]
                        for spk in range(self.num_spk)
                        if "dereverb{}".format(spk + 1) in mask_pre
                    ]
                    assert len(mask_wpe_pre) == dereverb_speech_ref.size(1), (
                        len(mask_wpe_pre),
                        dereverb_speech_ref.size(1),
                    )
                    dereverb_speech_ref = torch.unbind(dereverb_speech_ref, dim=1)
                    dereverb_spectrum_ref = [
                        ComplexTensor(*torch.unbind(self.enh_model.stft(dr)[0], dim=-1))
                        for dr in dereverb_speech_ref
                    ]
                    dereverb_mask_ref = self._create_mask_label(
                        spectrum_mix, dereverb_spectrum_ref, mask_type=self.mask_type
                    )

                    tf_dereverb_loss, perm_d = self._permutation_loss(
                        dereverb_mask_ref, mask_wpe_pre, loss_func
                    )
                    tf_loss = tf_loss + tf_dereverb_loss

                if "noise1" in mask_pre:
                    if noise_ref is None:
                        raise ValueError(
                            "No noise reference for training!\n"
                            'Please specify "--use_noise_ref true" in run.sh'
                        )

                    noise_ref = torch.unbind(noise_ref, dim=1)
                    noise_spectrum_ref = [
                        ComplexTensor(*torch.unbind(self.enh_model.stft(nr)[0], dim=-1))
                        for nr in noise_ref
                    ]
                    noise_mask_ref = self._create_mask_label(
                        spectrum_mix, noise_spectrum_ref, mask_type=self.mask_type
                    )

                    mask_noise_pre = [
                        mask_pre["noise{}".format(n + 1)]
                        for n in range(self.num_noise_type)
                    ]
                    tf_noise_loss, perm_n = self._permutation_loss(
                        noise_mask_ref, mask_noise_pre, loss_func
                    )
                    tf_loss = tf_loss + tf_noise_loss
            else:
                raise ValueError("Unsupported loss type: %s" % self.loss_type)

            loss = tf_loss
            loss = loss + loss_speech + loss_noise
            return loss, spectrum_pre, mask_pre, tf_length, perm, stats

        else:
            if speech_ref is not None and speech_ref.dim() == 4:
                # For si_snr loss of multi-channel input,
                # only select one channel as the reference
                speech_ref = speech_ref[..., self.ref_channel]

            if mode == -1:
                speech_pre, speech_lengths, *__ = self.enh_model.forward_rawwav(
                    speech_mix, text, speech_lengths, text_lengths
                )
            else:
                speech_pre, speech_lengths, _, latent, zlens = self.enh_model.forward_rawwav(
                    speech_mix, text, speech_lengths, text_lengths, return_latent=True
                )
                if mode == 1:
                    latent = latent.detach()
                category_pre = self.disc_model(latent, zlens)

            if not cal_loss:
                loss, perm = None, None
                return loss, speech_pre, None, speech_lengths, perm, stats


            # speech_pre: list[(batch, sample)]
            assert speech_pre[0].dim() == 2, speech_pre[0].dim()
            speech_ref = torch.unbind(speech_ref, dim=1)

            si_snr_loss, perm = self._permutation_loss(
                speech_ref, speech_pre, self.si_snr_loss_zeromean
            )
            # pdb.set_trace()
            if mode == 0:
                loss_real = ((category_pre[gan_ref == 1] - 1) ** 2).sum()
                loss_simu = ((category_pre[gan_ref == 0]) ** 2).sum()
                disc_loss = (loss_real + loss_simu) / category_pre.shape[0]
                loss = disc_loss + si_snr_loss
            elif mode == 1:
                loss_real = ((category_pre[gan_ref == 1] - 1) ** 2).sum()
                loss_simu = ((category_pre[gan_ref == 0]) ** 2).sum()
                disc_loss = (loss_real + loss_simu) / category_pre.shape[0]
                loss = disc_loss
            elif mode == -1:
                loss = si_snr_loss
            else:
                raise Exception(f"no supported mode with gan_ref: {mode}")

            # pdb.set_trace()
            loss = loss + loss_speech + loss_noise
            return loss, speech_pre, None, speech_lengths, perm, si_snr_loss, disc_loss, stats

    @staticmethod
    def tf_mse_loss(ref, inf):
        """time-frequency MSE loss.

        Args:
            ref: (Batch, T, F) or (Batch, T, C, F)
            inf: (Batch, T, F) or (Batch, T, C, F)
        Returns:
            loss: (Batch,)
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)
        diff = ref - inf
        if isinstance(diff, ComplexTensor):
            mseloss = diff.real ** 2 + diff.imag ** 2
        else:
            mseloss = diff ** 2
        if ref.dim() == 3:
            mseloss = mseloss.mean(dim=[1, 2])
        elif ref.dim() == 4:
            mseloss = mseloss.mean(dim=[1, 2, 3])
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )

        return mseloss

    @staticmethod
    def tf_log_mse_loss(ref, inf):
        """time-frequency log-MSE loss.

        Args:
            ref: (Batch, T, F) or (Batch, T, C, F)
            inf: (Batch, T, F) or (Batch, T, C, F)
        Returns:
            loss: (Batch,)
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)
        diff = ref - inf
        if isinstance(diff, ComplexTensor):
            log_mse_loss = diff.real ** 2 + diff.imag ** 2
        else:
            log_mse_loss = diff ** 2
        if ref.dim() == 3:
            log_mse_loss = torch.log10(log_mse_loss.sum(dim=[1, 2])) * 10
        elif ref.dim() == 4:
            log_mse_loss = torch.log10(log_mse_loss.sum(dim=[1, 2, 3])) * 10
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )

        return log_mse_loss

    @staticmethod
    def tf_l1_loss(ref, inf):
        """time-frequency L1 loss.

        Args:
            ref: (Batch, T, F) or (Batch, T, C, F)
            inf: (Batch, T, F) or (Batch, T, C, F)
        Returns:
            loss: (Batch,)
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)
        if isinstance(inf, ComplexTensor):
            l1loss = abs(ref - inf + 1e-15)
        else:
            l1loss = abs(ref - inf)
        if ref.dim() == 3:
            l1loss = l1loss.mean(dim=[1, 2])
        elif ref.dim() == 4:
            l1loss = l1loss.mean(dim=[1, 2, 3])
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )
        return l1loss

    @staticmethod
    def si_snr_loss(ref, inf):
        """SI-SNR loss

        Args:
            ref: (Batch, samples)
            inf: (Batch, samples)
        Returns:
            loss: (Batch,)
        """
        ref = ref / torch.norm(ref, p=2, dim=1, keepdim=True)
        inf = inf / torch.norm(inf, p=2, dim=1, keepdim=True)

        s_target = (ref * inf).sum(dim=1, keepdims=True) * ref
        e_noise = inf - s_target

        si_snr = 20 * torch.log10(
            torch.norm(s_target, p=2, dim=1) / torch.norm(e_noise, p=2, dim=1)
        )
        return -si_snr

    @staticmethod
    def si_snr_loss_zeromean(ref, inf):
        """SI-SNR loss with zero-mean in pre-processing.

        Args:
            ref: (Batch, samples)
            inf: (Batch, samples)
        Returns:
            loss: (Batch,)
        """
        eps = 1e-8

        assert ref.size() == inf.size()
        B, T = ref.size()
        # mask padding position along T

        # Step 1. Zero-mean norm
        mean_target = torch.sum(ref, dim=1, keepdim=True) / T
        mean_estimate = torch.sum(inf, dim=1, keepdim=True) / T
        zero_mean_target = ref - mean_target
        zero_mean_estimate = inf - mean_estimate

        # Step 2. SI-SNR with order
        # reshape to use broadcast
        s_target = zero_mean_target  # [B, T]
        s_estimate = zero_mean_estimate  # [B, T]
        # s_target = <s', s>s / ||s||^2
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=1, keepdim=True)  # [B, 1]
        s_target_energy = torch.sum(s_target ** 2, dim=1, keepdim=True) + eps  # [B, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, T]
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [B, T]

        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=1) / (
            torch.sum(e_noise ** 2, dim=1) + eps
        )
        # print('pair_si_snr',pair_wise_si_snr[0,:])
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + eps)  # [B]
        # print(pair_wise_si_snr)

        return -1 * pair_wise_si_snr

    @staticmethod
    def _permutation_loss(ref, inf, criterion, perm=None):
        """The basic permutation loss function.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            inf (List[torch.Tensor]): [(batch, ...), ...]
            criterion (function): Loss function
            perm (torch.Tensor): specified permutation (batch, num_spk)
        Returns:
            loss (torch.Tensor): (batch)
            perm (torch.Tensor): (batch, num_spk)
                                 e.g. tensor([[1, 0, 2], [0, 1, 2]])
        """
        assert len(ref) == len(inf), (len(ref), len(inf))
        num_spk = len(ref)

        def pair_loss(permutation):
            return sum(
                [criterion(ref[s], inf[t]) for s, t in enumerate(permutation)]
            ) / len(permutation)

        if perm is None:
            device = ref[0].device
            all_permutations = list(permutations(range(num_spk)))
            losses = torch.stack([pair_loss(p) for p in all_permutations], dim=1)
            loss, perm = torch.min(losses, dim=1)
            perm = torch.index_select(
                torch.tensor(all_permutations, device=device, dtype=torch.long),
                0,
                perm,
            )
        else:
            loss = torch.tensor(
                [
                    torch.tensor(
                        [
                            criterion(
                                ref[s][batch].unsqueeze(0), inf[t][batch].unsqueeze(0)
                            )
                            for s, t in enumerate(p)
                        ]
                    ).mean()
                    for batch, p in enumerate(perm)
                ]
            )

        return loss.mean(), perm

    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]

        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
