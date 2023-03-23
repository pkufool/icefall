#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=15
stage=-1
stop_stage=11

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/aishell
#      You can find data_aishell, resource_aishell inside it.
#      You can download them from https://www.openslr.org/33
#
#  - $dl_dir/lm
#      This directory contains the language model downloaded from
#        https://huggingface.co/pkufool/aishell_lm
#
#        - 3-gram.unpruned.arpa
#
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech

dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

# vocab size for sentence piece models.
# It will generate data/lang_bbpe_xxx,
# data/lang_bbpe_yyy if the array contains xxx, yyy
vocab_sizes=(
  2000
  1000
  500
)

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  log "stage -1: Download LM"
  # We assume that you have installed the git-lfs, if not, you could install it
  # using: `sudo apt-get install git-lfs && git-lfs install`
  git lfs 1>/dev/null 2>&1 || (echo "please install git-lfs, consider using: sudo apt-get install git-lfs && git-lfs install" && exit 1)

  if [ ! -f $dl_dir/lm/3-gram.unpruned.arpa ]; then
    git clone https://huggingface.co/pkufool/aishell_lm $dl_dir/lm
    pushd $dl_dir/lm
    git lfs pull --include "3-gram.unpruned.arpa"
    popd
  fi
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "stage 0: Download data"

  # If you have pre-downloaded it to /path/to/aishell,
  # you can create a symlink
  #
  #   ln -sfv /path/to/aishell $dl_dir/aishell
  #
  # The directory structure is
  # aishell/
  # |-- data_aishell
  # |   |-- transcript
  # |   `-- wav
  # `-- resource_aishell
  #     |-- lexicon.txt
  #     `-- speaker.info

  if [ ! -d $dl_dir/aishell/data_aishell/wav/train ]; then
    lhotse download aishell $dl_dir
  fi

  # If you have pre-downloaded it to /path/to/musan,
  # you can create a symlink
  #
  #   ln -sfv /path/to/musan $dl_dir/musan
  #
  if [ ! -d $dl_dir/musan ]; then
    lhotse download musan $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare aishell manifest"
  # We assume that you have downloaded the aishell corpus
  # to $dl_dir/aishell
  if [ ! -f data/manifests/.aishell_manifests.done ]; then
    mkdir -p data/manifests
    lhotse prepare aishell $dl_dir/aishell data/manifests
    touch data/manifests/.aishell_manifests.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to data/musan
  if [ ! -f data/manifests/.musan_manifests.done ]; then
    log "It may take 6 minutes"
    mkdir -p data/manifests
    lhotse prepare musan $dl_dir/musan data/manifests
    touch data/manifests/.musan_manifests.done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute fbank for aishell"
  if [ ! -f data/fbank/.aishell.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_aishell.py
    touch data/fbank/.aishell.done
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute fbank for musan"
  if [ ! -f data/fbank/.msuan.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_musan.py
    touch data/fbank/.msuan.done
  fi
fi

lang_phone_dir=data/lang_phone
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Prepare phone based lang"
  mkdir -p $lang_phone_dir

  (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; ) |
    cat - $dl_dir/aishell/resource_aishell/lexicon.txt |
    sort | uniq > $lang_phone_dir/lexicon.txt

  ./local/generate_unique_lexicon.py --lang-dir $lang_phone_dir

  if [ ! -f $lang_phone_dir/L_disambig.pt ]; then
    ./local/prepare_lang.py --lang-dir $lang_phone_dir
  fi

  # Train a bigram P for MMI training
  if [ ! -f $lang_phone_dir/transcript_words.txt ]; then
    log "Generate data to train phone based bigram P"
    aishell_text=$dl_dir/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt
    aishell_train_uid=$dl_dir/aishell/data_aishell/transcript/aishell_train_uid
    find $dl_dir/aishell/data_aishell/wav/train -name "*.wav" | sed 's/\.wav//g' | awk -F '/' '{print $NF}' > $aishell_train_uid
    awk 'NR==FNR{uid[$1]=$1} NR!=FNR{if($1 in uid) print $0}' $aishell_train_uid $aishell_text |
	    cut -d " " -f 2- > $lang_phone_dir/transcript_words.txt
  fi

  if [ ! -f $lang_phone_dir/transcript_tokens.txt ]; then
    ./local/convert_transcript_words_to_tokens.py \
      --lexicon $lang_phone_dir/uniq_lexicon.txt \
      --transcript $lang_phone_dir/transcript_words.txt \
      --oov "<UNK>" \
      > $lang_phone_dir/transcript_tokens.txt
  fi

  if [ ! -f $lang_phone_dir/P.arpa ]; then
    ./shared/make_kn_lm.py \
      -ngram-order 2 \
      -text $lang_phone_dir/transcript_tokens.txt \
      -lm $lang_phone_dir/P.arpa
  fi

  if [ ! -f $lang_phone_dir/P.fst.txt ]; then
    python3 -m kaldilm \
      --read-symbol-table="$lang_phone_dir/tokens.txt" \
      --disambig-symbol='#0' \
      --max-order=2 \
      $lang_phone_dir/P.arpa > $lang_phone_dir/P.fst.txt
  fi
fi

lang_char_dir=data/lang_char
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Prepare char based lang"
  mkdir -p $lang_char_dir
  # We reuse words.txt from phone based lexicon
  # so that the two can share G.pt later.
  cp $lang_phone_dir/words.txt $lang_char_dir

  cat $dl_dir/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt |
  cut -d " " -f 2- | sed -e 's/[ \t\r\n]*//g' > $lang_char_dir/text

  if [ ! -f $lang_char_dir/L_disambig.pt ]; then
    ./local/prepare_char.py
  fi
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Prepare Byte BPE based lang"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bbpe_${vocab_size}
    mkdir -p $lang_dir
    # We reuse words.txt from phone based lexicon
    # so that the two can share G.pt later.
    cp $lang_phone_dir/words.txt $lang_dir

    cat $dl_dir/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt |
    cut -d " " -f 2- | sed -e 's/^[ \t]*//;s/[ \t]*$//' > $lang_dir/text

    if [ ! -f $lang_dir/bbpe.model ]; then
      ./local/train_bbpe_model.py \
        --lang-dir $lang_dir \
        --vocab-size $vocab_size \
        --transcript $lang_dir/text
    fi
  done
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Prepare G"
  # We assume you have install kaldilm, if not, please install
  # it using: pip install kaldilm

  mkdir -p data/lm
  if [ ! -f data/lm/G_3_gram.fst.txt ]; then
    # It is used in building HLG
    python3 -m kaldilm \
      --read-symbol-table="$lang_phone_dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      $dl_dir/lm/3-gram.unpruned.arpa > data/lm/G_3_gram.fst.txt
  fi
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  log "Stage 8: Compile HLG"
  ./local/compile_hlg.py --lang-dir $lang_phone_dir
  ./local/compile_hlg.py --lang-dir $lang_char_dir
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Generate LM training data"

  log "Processing char based data"
  out_dir=data/lm_training_char
  mkdir -p $out_dir $dl_dir/lm

  if [ ! -f $dl_dir/lm/aishell-train-word.txt ]; then
    cp $lang_phone_dir/transcript_words.txt $dl_dir/lm/aishell-train-word.txt
  fi

  ./local/prepare_char_lm_training_data.py \
    --lang-char data/lang_char \
    --lm-data $dl_dir/lm/aishell-train-word.txt \
    --lm-archive $out_dir/lm_data.pt

  if [ ! -f $dl_dir/lm/aishell-valid-word.txt ]; then
    aishell_text=$dl_dir/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt
    aishell_valid_uid=$dl_dir/aishell/data_aishell/transcript/aishell_valid_uid
    find $dl_dir/aishell/data_aishell/wav/dev -name "*.wav" | sed 's/\.wav//g' | awk -F '/' '{print $NF}' > $aishell_valid_uid
    awk 'NR==FNR{uid[$1]=$1} NR!=FNR{if($1 in uid) print $0}' $aishell_valid_uid $aishell_text |
	    cut -d " " -f 2- > $dl_dir/lm/aishell-valid-word.txt
  fi

  ./local/prepare_char_lm_training_data.py \
    --lang-char data/lang_char \
    --lm-data $dl_dir/lm/aishell-valid-word.txt \
    --lm-archive $out_dir/lm_data_valid.pt

  if [ ! -f $dl_dir/lm/aishell-test-word.txt ]; then
    aishell_text=$dl_dir/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt
    aishell_test_uid=$dl_dir/aishell/data_aishell/transcript/aishell_test_uid
    find $dl_dir/aishell/data_aishell/wav/test -name "*.wav" | sed 's/\.wav//g' | awk -F '/' '{print $NF}' > $aishell_test_uid
    awk 'NR==FNR{uid[$1]=$1} NR!=FNR{if($1 in uid) print $0}' $aishell_test_uid $aishell_text |
	    cut -d " " -f 2- > $dl_dir/lm/aishell-test-word.txt
  fi

  ./local/prepare_char_lm_training_data.py \
    --lang-char data/lang_char \
    --lm-data $dl_dir/lm/aishell-test-word.txt \
    --lm-archive $out_dir/lm_data_test.pt
fi


if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
  log "Stage 10: Sort LM training data"
  # Sort LM training data by sentence length in descending order
  # for ease of training.
  #
  # Sentence length equals to the number of tokens
  # in a sentence.

  out_dir=data/lm_training_char
  mkdir -p $out_dir
  ln -snf ../../../librispeech/ASR/local/sort_lm_training_data.py local/
  
  ./local/sort_lm_training_data.py \
    --in-lm-data $out_dir/lm_data.pt \
    --out-lm-data $out_dir/sorted_lm_data.pt \
    --out-statistics $out_dir/statistics.txt

  ./local/sort_lm_training_data.py \
    --in-lm-data $out_dir/lm_data_valid.pt \
    --out-lm-data $out_dir/sorted_lm_data-valid.pt \
    --out-statistics $out_dir/statistics-valid.txt

  ./local/sort_lm_training_data.py \
    --in-lm-data $out_dir/lm_data_test.pt \
    --out-lm-data $out_dir/sorted_lm_data-test.pt \
    --out-statistics $out_dir/statistics-test.txt
fi

if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
  log "Stage 11: Train RNN LM model"
  python ../../../icefall/rnn_lm/train.py \
    --start-epoch 0 \
    --world-size 1 \
    --num-epochs 20 \
    --use-fp16 0 \
    --embedding-dim 512 \
    --hidden-dim 512 \
    --num-layers 2 \
    --batch-size 400 \
    --exp-dir rnnlm_char/exp \
    --lm-data data/lm_training_char/sorted_lm_data.pt \
    --lm-data-valid data/lm_training_char/sorted_lm_data-valid.pt \
    --vocab-size 4336 \
    --master-port 12345
fi
