# Copyright      2023-2024  Xiaomi Corp.        (authors: Zengwei Yao
#                                                         Han Zhu
#                                                         Wei Kang)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Dict, List

from text_normalizer import custom_english_cleaners

try:
    import sentencepiece as spm
except Exception as ex:
    raise RuntimeError(f"{ex}\nPlease run\npip install sentencepiece")

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )


class Tokenizer(object):
    def __init__(self, token_type: str, token_file: str):
        """
        Args:
          token_type: the type of tokenizer, e.g., bpe, letter, phone.
          token_file: the file that contains information that maps tokens to ids,
            which is a text file with '{token} {token_id}' per line if type is
            char or phone, otherwise it is a bpe_model file.
        """
        self.token_type = token_type
        assert token_type in ["bpe", "letter", "phone"], token_type

        if token_type == "bpe":
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(token_file)
            self.token2id: Dict[str, int] = {
                self.sp.id_to_piece(i): i for i in range(self.sp.vocab_size())
            }
            self.pad_id = self.token2id["<pad>"]
            self.sos_id = self.token2id["<s>"]  # beginning of an utterance (bos)
            self.eos_id = self.token2id["</s>"]  # end of an utterance (eos)
            self.space_id = self.token2id["<blk>"]  # word separator (whitespace)
            self.vocab_size = self.sp.get_piece_size()
        else:
            self.token2id: Dict[str, int] = {}
            with open(token_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    info = line.rstrip().split()
                    if len(info) == 1:
                        # case of space
                        token = " "
                        id = int(info[0])
                    else:
                        token, id = info[0], int(info[1])
                    assert token not in self.token2id, token
                    self.token2id[token] = id
            # Refer to https://github.com/rhasspy/piper/blob/master/TRAINING.md
            self.pad_id = self.token2id["_"]  # padding
            self.sos_id = self.token2id["^"]  # beginning of an utterance (bos)
            self.eos_id = self.token2id["$"]  # end of an utterance (eos)
            self.space_id = self.token2id[" "]  # word separator (whitespace)

            self.vocab_size = len(self.token2id)

    def texts_to_token_ids(
        self,
        texts: List[str],
        intersperse_blank: bool = False,
        return_tokens: bool = False,
        lang: str = "en-us",
    ) -> List[List[int]]:
        """
        Args:
          texts:
            A list of transcripts.
          intersperse_blank:
            Whether to intersperse blanks in the token sequence.
            Used when alignment is from MAS.
          lang:
            Language argument passed to phonemize_espeak().

        Returns:
          Return a list of token id list [utterance][token_id]
        """
        for i in range(len(texts)):
            # Text normalization
            texts[i] = custom_english_cleaners(texts[i])

        tokens_list = []
        token_ids_list = []

        if self.token_type == "bpe":
            token_ids_list = self.sp.encode(texts)
            if return_tokens:
                tokens_list = self.sp.encode(texts, out_type=str)

        elif self.token_type == "phone":
            for text in texts:
                raw_tokens_list = phonemize_espeak(text, lang)
                tokens = []
                for t in raw_tokens_list:
                    tokens.extend(t)
                token_ids = []
                for t in tokens:
                    if t not in self.token2id:
                        logging.warning(f"Skip OOV {t}")
                        continue
                    token_ids.append(self.token2id[t])
                if return_tokens:
                    tokens_list.append(tokens)
                token_ids_list.append(token_ids)
        else:
            for text in texts:
                token_ids = []
                tokens = []
                for t in text:
                    if t not in self.token2id:
                        logging.warning(f"Skip OOV {t}")
                        continue
                    token_ids.append(self.token2id[t])
                    tokens.append(t)

                token_ids_list.append(token_ids)
                if return_tokens:
                    tokens_list.append(tokens)

        if intersperse_blank:
            for i in range(len(token_ids_list)):
                token_ids_list[i] = intersperse(token_ids_list[i], self.pad_id)

        if return_tokens:
            return token_ids_list, tokens_list

        return token_ids_list

    def tokens_to_token_ids(
        self,
        tokens_list: List[List[str]],
        intersperse_blank: bool = False,
    ) -> List[List[int]]:
        """
        Args:
          tokens_list:
            A list of token list, each corresponding to one utterance.
          intersperse_blank:
            Whether to intersperse blanks in the token sequence.

        Returns:
          Return a list of token id list [utterance][token_id]
        """
        token_ids_list = []

        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logging.warning(f"Skip OOV {t}")
                    continue
                token_ids.append(self.token2id[t])

            if intersperse_blank:
                token_ids = intersperse(token_ids, self.pad_id)

            token_ids_list.append(token_ids)

        return token_ids_list


def intersperse(sequence, item=0):
    """
    Intersperse a sequence with a given item.

    If the input sequence is [1, 2, 3] and the item is 0, the output will be
    [0, 1, 0, 2, 0, 3, 0].
    """
    result = [item] * (len(sequence) * 2 + 1)
    result[1::2] = sequence
    return result
