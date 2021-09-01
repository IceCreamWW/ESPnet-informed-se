from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union

from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer


class RawTokenizer(AbsTokenizer):
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f")"
        )

    def text2tokens(self, line: str) -> List[str]:
        return line.strip().split()

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return " ".join(tokens)
