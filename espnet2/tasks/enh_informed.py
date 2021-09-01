import argparse
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.decoder.null_decoder import NullDecoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.informed_encoder.abs_informed_encoder import AbsInformedEncoder
from espnet2.enh.informed_encoder.text_encoder import EmbeddingEncoder as EmbeddingTextEncoder
from espnet2.enh.informed_encoder.text_encoder import RNNEncoder as RNNTextEncoder
from espnet2.enh.informed_encoder.text_encoder import DPRNNEncoder as DPRNNTextEncoder
from espnet2.enh.informed_encoder.text_encoder import TactronEncoder as TactronTextEncoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.encoder.null_encoder import NullEncoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.fusion.abs_fusion import AbsFusion
from espnet2.enh.fusion.transformer_fusion import TransformerFusion, ASRDecoderFusion
from espnet2.enh.fusion.concat_fusion import ConcatFusion
from espnet2.enh.espnet_enh_informed_model import ESPnetEnhancementInformedModel
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.separator.asteroid_models import AsteroidModel_Converter
from espnet2.enh.separator.conformer_separator import ConformerSeparator
from espnet2.enh.separator.dprnn_separator import DPRNNSeparator
from espnet2.enh.separator.neural_beamformer import NeuralBeamformer
from espnet2.enh.separator.rnn_separator import RNNSeparator
from espnet2.enh.separator.tcn_separator import TCNSeparator
from espnet2.enh.separator.transformer_separator import TransformerSeparator
from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor, CommonPreprocessor_multi
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none
from espnet2.text.utils import *

import pdb

informed_encoder_choices = ClassChoices(
    name="informed_encoder",
    classes=dict(embedding=EmbeddingTextEncoder, dprnn=DPRNNTextEncoder, rnn=RNNTextEncoder, tactron=TactronTextEncoder),
    type_check=AbsInformedEncoder,
    default="embedding",
)

encoder_choices = ClassChoices(
    name="encoder",
    classes=dict(stft=STFTEncoder, conv=ConvEncoder, same=NullEncoder),
    type_check=AbsEncoder,
    default="stft",
)


fusion_choices = ClassChoices(
    name="fusion",
    classes=dict(transformer=TransformerFusion, asrdecoder=ASRDecoderFusion, concat=ConcatFusion),
    type_check=AbsFusion,
    default="transformer",
)

separator_choices = ClassChoices(
    name="separator",
    classes=dict(
        rnn=RNNSeparator,
        tcn=TCNSeparator,
        dprnn=DPRNNSeparator,
        transformer=TransformerSeparator,
        conformer=ConformerSeparator,
        wpe_beamformer=NeuralBeamformer,
        asteroid=AsteroidModel_Converter,
    ),
    type_check=AbsSeparator,
    default="rnn",
)

decoder_choices = ClassChoices(
    name="decoder",
    classes=dict(stft=STFTDecoder, conv=ConvDecoder, same=NullDecoder),
    type_check=AbsDecoder,
    default="stft",
)

MAX_REFERENCE_NUM = 100


class EnhancementInformedTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    class_choices_list = [
        # --informed_encoder and --informed_encoder_conf
        informed_encoder_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --fusion and --fusion_conf
        fusion_choices,
        # --separator and --separator_conf
        separator_choices,
        # --decoder and --decoder_conf
        decoder_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        # required = parser.get_default("required")

        required = parser.get_default("required")

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )

        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetEnhancementInformedModel),
            help="The keyword arguments for model class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=False,
            help="Apply preprocessing to data or not",
        )

        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["raw", "bpe", "char", "word", "phn"],
            help="The text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        parser.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Apply text cleaning",
        )
        parser.add_argument(
            "--g2p",
            type=str_or_none,
            choices=[None, "g2p_en", "g2p_en_no_space", "pyopenjtalk", "pyopenjtalk_kana"],
            default=None,
            help="Specify g2p method if --token_type=phn",
        )

        group.add_argument(
            "--partially_informed",
            type=str2bool,
            default=False,
            help="apply partially informed training or not",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()

        return CommonCollateFn(float_pad_value=0.0, int_pad_value=0)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            retval = CommonPreprocessor_multi(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                text_name=["informed", "informed_ali_ref"],
                g2p_type=args.g2p,
                unk_symbol='UNK',
                processor=None if not args.partially_informed else partially_informed_processor
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech_mix", "speech_ref1", "informed")
        else:
            # Recognition mode
            retval = ("speech_mix", "informed")
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ["dereverb_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval += ["speech_ref{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval += ["noise_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval += ["informed_ali_ref"]
        retval += ["informed_text_raw"]
        retval = tuple(retval)
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetEnhancementInformedModel:
        assert check_argument_types()

        informed_encoder = informed_encoder_choices.get_class(args.informed_encoder)(**args.informed_encoder_conf)
        encoder = encoder_choices.get_class(args.encoder)(**args.encoder_conf)
        fusion = fusion_choices.get_class(args.fusion)(**args.fusion_conf)
        separator = separator_choices.get_class(args.separator)(**args.separator_conf)
        decoder = decoder_choices.get_class(args.decoder)(**args.decoder_conf)

        # 1. Build model
        model = ESPnetEnhancementInformedModel(
            informed_encoder=informed_encoder, fusion=fusion,
            encoder=encoder, separator=separator, decoder=decoder, **args.model_conf
        )

        # FIXME(kamo): Should be done in model?
        # 2. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model

import logging

def partially_informed_processor(preprocessor, uid, data):
    # pdb.set_trace()
    informed_ali_ref = data["informed_ali_ref"]
    iepoch = data['iepoch']
    percent = 0.1

    informed_ali_ctm = ali2ctm(informed_ali_ref)
    informed_ali_ctm = recompute_ctm(informed_ali_ctm, 400, 160, 20, 10)
    informed_ali_seq = [t[0] for t in informed_ali_ctm]
    num_phns = int(min(max(len(informed_ali_seq) * percent, 5), len(informed_ali_seq)))
    start = random.randint(0, len(informed_ali_seq) - num_phns)
    occurences = [(informed_ali_ctm[start][1], informed_ali_ctm[start + num_phns - 1][2])]

    data["speech_ref1"] = mask_occurences_in_sequence(data["speech_ref1"], occurences)
    data["informed"] = np.array(informed_ali_seq[start:start+num_phns])

    del data["informed_text_raw"]
    return data
# 
# TODO: debug this function
# def partially_informed_processor(preprocessor, uid, data):
# 
#     informed_words = data["informed_text_raw"].strip().split()
#     informed_ali_ref = data["informed_ali_ref"]
#     iepoch = data['iepoch']
#     if not hasattr(preprocessor, "parital_tokenizer"):
#         preprocessor.partial_tokenizer = PhonemeTokenizer(g2p_type="g2p_en_no_space")
# 
#     tokenizer = preprocessor.partial_tokenizer
#     token_id_converter = preprocessor.token_id_converter
#     percent = iepoch * 4  / 100
# 
#     informed_ali_ctm = ali2ctm(informed_ali_ref)
#     informed_ali_ctm = recompute_ctm(informed_ali_ctm, 400, 160, 20, 10)
#     num_words = int(min(max(len(informed_words) * percent, 1), len(informed_words)))
# 
#     for i in range(3):
#         start = random.randint(0, len(informed_words) - num_words)
#         partial_text = " ".join(informed_words[start:start + num_words])
#         partial_ids = token_id_converter.tokens2ids(tokenizer.text2tokens(partial_text))
#         if len(partial_ids) > 2:
#             break
#         else:
#             num_words = min(num_words + 1, len(informed_words))
# 
#     if len(partial_ids) > 2:
#         informed_ali_seq = [t[0] for t in informed_ali_ctm]
#         occurences = find_occurences(informed_ali_seq, partial_ids, informed_ali_ctm)
#         if len(occurences) == 0:
#             logging.info(f"uid={uid}, partial_text={partial_text},partial_ids={partial_ids},seq={informed_ali_seq}")
#         else:
#             data["speech_ref1"] = mask_occurences_in_sequence(data["speech_ref1"], occurences)
#             data["informed"] = np.array(partial_ids)
# 
#     del data["informed_text_raw"]
# 
#     return data
# 
