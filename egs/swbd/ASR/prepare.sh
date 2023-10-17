#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=15
stage=-1
stop_stage=100

# We assume dl_dir (download dir) contains the following
# directories and files. Most of them can't be downloaded automatically
# as they are not publically available and require a license purchased
# from the LDC.
#
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech

dl_dir=./download
# swbd1_dir="/export/corpora3/LDC/LDC97S62"
swbd1_dir=./download/LDC97S62/

# eval2000_dir contains the following files and directories
# downloaded from LDC website:
#  - LDC2002S09
#       - hub5e_00
#  - LDC2002T43
#       - reference
eval2000_dir="/export/corpora2/LDC/eval2000"

rt03_dir="/export/corpora/LDC/LDC2007S10"
fisher_dir="/export/corpora3/LDC/LDC2004T19"

. shared/parse_options.sh || exit 1

# vocab size for sentence piece models.
# It will generate data/lang_bpe_xxx,
# data/lang_bpe_yyy if the array contains xxx, yyy
vocab_sizes=(
    # 5000
    # 2000
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

log "swbd1_dir: $swbd1_dir"
log "eval2000_dir: $eval2000_dir"
log "rt03_dir: $rt03_dir"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Prepare SwitchBoard manifest"
    # We assume that you have downloaded the SwitchBoard corpus
    # to respective dirs
    mkdir -p data/manifests
    if [ ! -e data/manifests/.swbd.done ]; then
        lhotse prepare switchboard --absolute-paths 1 --omit-silence $swbd1_dir data/manifests/swbd
        ./local/normalize_and_filter_supervisions.py \
            data/manifests/swbd/swbd_supervisions_all.jsonl.gz \
            data/manifests/swbd/swbd_supervisions_all_norm.jsonl.gz
        mv data/manifests/swbd/swbd_supervisions_all.jsonl.gz data/manifests/swbd/swbd_supervisions_orig.jsonl.gz
        mv data/manifests/swbd/swbd_supervisions_all_norm.jsonl.gz data/manifests/swbd/swbd_supervisions_all.jsonl.gz

        lhotse cut simple \
            -r data/manifests/swbd/swbd_recordings_all.jsonl.gz \
            -s data/manifests/swbd/swbd_supervisions_all.jsonl.gz \
            data/manifests/swbd/swbd_train_all.jsonl.gz
        lhotse cut trim-to-supervisions \
            --discard-overlapping \
            --discard-extra-channels \
            data/manifests/swbd/swbd_train_all.jsonl.gz \
            data/manifests/swbd/swbd_train_all_trimmed.jsonl.gz

        num_splits=16
        mkdir -p data/manifests/swbd_split${num_splits}
        lhotse split ${num_splits} \
            data/manifests/swbd/swbd_train_all_trimmed.jsonl.gz \
            data/manifests/swbd_split${num_splits}

        lhotse prepare eval2000 --absolute-paths 1 $eval2000_dir data/manifests/eval2000
        ./local/normalize_eval2000.py \
            data/manifests/eval2000/eval2000_supervisions_unnorm.jsonl.gz \
            data/manifests/eval2000/eval2000_supervisions_all.jsonl.gz
        
        lhotse cut simple \
            -r data/manifests/eval2000/eval2000_recordings_all.jsonl.gz \
            -s data/manifests/eval2000/eval2000_supervisions_all.jsonl.gz \
            data/manifests/eval2000/eval2000_cuts_all.jsonl.gz

        lhotse cut trim-to-supervisions \
            --discard-overlapping \
            --discard-extra-channels \
            data/manifests/eval2000/eval2000_cuts_all.jsonl.gz \
            data/manifests/eval2000/eval2000_cuts_all_trimmed.jsonl.gz

        sed -e 's:((:(:' -e 's:<B_ASIDE>::g' -e 's:<E_ASIDE>::g' \
            $eval2000_dir/LDC2002T43/reference/hub5e00.english.000405.stm >  data/manifests/eval2000/stm
        cp $eval2000_dir/LDC2002T43/reference/en20000405_hub5.glm  $dir/glm

        # ./local/rt03_data_prep.sh $rt03_dir

        # normalize eval2000 and rt03 texts by
        # 1) convert upper to lower
        # 2) remove tags (%AH) (%HESITATION) (%UH)
        # 3) remove <B_ASIDE> <E_ASIDE>
        # 4) remove "(" or ")"
        # for x in  rt03; do
        #     cp data/local/${x}/text data/local/${x}/text.org
        #     paste -d "" \
        #         <(cut -f 1 -d" " data/local/${x}/text.org) \
        #         <(awk '{$1=""; print tolower($0)}' data/local/${x}/text.org | perl -pe 's| \(\%.*\)||g' | perl -pe 's| \<.*\>||g' | sed -e "s/(//g" -e "s/)//g") |
        #         sed -e 's/\s\+/ /g' >data/local/${x}/text
        #     rm data/local/${x}/text.org
        # done

        # lhotse fix data/manifests_rt03/swbd_recordings_rt03.jsonl.gz data/manifests_rt03/swbd_supervisions_rt03.jsonl.gz data/manifests

        touch data/manifests/.swbd.done
    fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Prepare musan manifest"
    # We assume that you have downloaded the musan corpus
    # to $dl_dir/musan
    mkdir -p data/manifests
    if [ ! -e data/manifests/.musan.done ]; then
        lhotse prepare musan $dl_dir/musan data/manifests
        touch data/manifests/.musan.done
    fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3 I: Compute fbank for SwitchBoard"
    if [ ! -e data/fbank/.swbd.done ]; then
        mkdir -p data/fbank/swbd_split${num_splits}/
        for index in $(seq 1 16); do
            ./local/compute_fbank_swbd.py --split-index ${index} &
        done
        wait
        pieces=$(find data/fbank/swbd_split${num_splits} -name "swbd_cuts_all.*.jsonl.gz")
        lhotse combine $pieces data/fbank/swbd_cuts_all.jsonl.gz
        touch data/fbank/.swbd.done
    fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3 II: Compute fbank for eval2000"
    if [ ! -e data/fbank/.eval2000.done ]; then
        mkdir -p data/fbank/eval2000/
        ./local/compute_fbank_eval2000.py 
        touch data/fbank/.eval2000.done
    fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 4: Compute fbank for musan"
    mkdir -p data/fbank
    if [ ! -e data/fbank/.musan.done ]; then
        ./local/compute_fbank_musan.py
        touch data/fbank/.musan.done
    fi
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Stage 5: Prepare phone based lang"
    lang_dir=data/lang_phone
    mkdir -p $lang_dir

    if ! which jq; then
      echo "This script is intended to be used with jq but you have not installed jq
      Note: in Linux, you can install jq with the following command:
      1. wget -O jq https://github.com/stedolan/jq/releases/download/jq-1.6/jq-linux64
      2. chmod +x ./jq
      3. cp jq /usr/bin" && exit 1
    fi
    if [ ! -f $lang_dir/text ] || [ ! -s $lang_dir/text ]; then
        log "Prepare text."
        gunzip -c data/manifests/swbd/swbd_supervisions_all.jsonl.gz \
        | jq '.text' | sed 's/"//g'  > $lang_dir/text
    fi

    log "Prepare dict"
    ./local/swbd1_prepare_dict.sh 
    cut -f 2- -d" " $lang_dir/text >${lang_dir}/input.txt
    # [noise] nsn
    # !sil sil
    # <unk> spn
    cat data/local/dict_nosp/lexicon.txt | sed 's/-//g' | sed 's/\[vocalizednoise\]/\[vocalized-noise\]/g' |
        sort | uniq >$lang_dir/lexicon_lower.txt

    cat $lang_dir/lexicon_lower.txt | tr a-z A-Z > $lang_dir/lexicon.txt

    if [ ! -f $lang_dir/L_disambig.pt ]; then
        ./local/prepare_lang.py --lang-dir $lang_dir
    fi

    if [ ! -f $lang_dir/L.fst ]; then
        log "Converting L.pt to L.fst"
        ./shared/convert-k2-to-openfst.py \
            --olabels aux_labels \
            $lang_dir/L.pt \
            $lang_dir/L.fst
    fi

    if [ ! -f $lang_dir/L_disambig.fst ]; then
        log "Converting L_disambig.pt to L_disambig.fst"
        ./shared/convert-k2-to-openfst.py \
            --olabels aux_labels \
            $lang_dir/L_disambig.pt \
            $lang_dir/L_disambig.fst
    fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    log "Stage 6: Prepare BPE based lang"

    for vocab_size in ${vocab_sizes[@]}; do
        lang_dir=data/lang_bpe_${vocab_size}
        mkdir -p $lang_dir
        # We reuse words.txt from phone based lexicon
        # so that the two can share G.pt later.
        cp data/lang_phone/words.txt $lang_dir

        if [ ! -f $lang_dir/transcript_words.txt ]; then
            log "Generate data for BPE training"

            cat data/lang_phone/text | cut -d " " -f 2- >$lang_dir/transcript_words.txt
        fi

        if [ ! -f $lang_dir/bpe.model ]; then
            ./local/train_bpe_model.py \
                --lang-dir $lang_dir \
                --vocab-size $vocab_size \
                --transcript $lang_dir/transcript_words.txt
        fi

        if [ ! -f $lang_dir/L_disambig.pt ]; then
            ./local/prepare_lang_bpe.py --lang-dir $lang_dir

            log "Validating $lang_dir/lexicon.txt"
            ./local/validate_bpe_lexicon.py \
                --lexicon $lang_dir/lexicon.txt \
                --bpe-model $lang_dir/bpe.model
        fi

        if [ ! -f $lang_dir/L.fst ]; then
            log "Converting L.pt to L.fst"
            ./shared/convert-k2-to-openfst.py \
                --olabels aux_labels \
                $lang_dir/L.pt \
                $lang_dir/L.fst
        fi

        if [ ! -f $lang_dir/L_disambig.fst ]; then
            log "Converting L_disambig.pt to L_disambig.fst"
            ./shared/convert-k2-to-openfst.py \
                --olabels aux_labels \
                $lang_dir/L_disambig.pt \
                $lang_dir/L_disambig.fst
        fi
    done
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    log "Stage 7: Prepare bigram token-level P for MMI training"

    for vocab_size in ${vocab_sizes[@]}; do
        lang_dir=data/lang_bpe_${vocab_size}

        if [ ! -f $lang_dir/transcript_tokens.txt ]; then
            ./local/convert_transcript_words_to_tokens.py \
                --lexicon $lang_dir/lexicon.txt \
                --transcript $lang_dir/transcript_words.txt \
                --oov "<UNK>" \
                >$lang_dir/transcript_tokens.txt
        fi

        if [ ! -f $lang_dir/P.arpa ]; then
            ./shared/make_kn_lm.py \
                -ngram-order 2 \
                -text $lang_dir/transcript_tokens.txt \
                -lm $lang_dir/P.arpa
        fi

        if [ ! -f $lang_dir/P.fst.txt ]; then
            python3 -m kaldilm \
                --read-symbol-table="$lang_dir/tokens.txt" \
                --disambig-symbol='#0' \
                --max-order=2 \
                $lang_dir/P.arpa >$lang_dir/P.fst.txt
        fi
    done
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    log "Stage 8: Prepare G"
    lang_dir=data/lang_phone
    # We assume you have install kaldilm, if not, please install
    # it using: pip install kaldilm

    mkdir -p data/lm
    if [ ! -f data/lm/G_3_gram.fst.txt ]; then
        # It is used in building HLG
        ./shared/make_kn_lm.py \
            -ngram-order 3 \
            -text ${lang_dir}/input.txt \
            -lm data/lm/3-gram.arpa
        python3 -m kaldilm \
            --read-symbol-table="data/lang_phone/words.txt" \
            --disambig-symbol='#0' \
            --max-order=3 \
            data/lm/3-gram.arpa >data/lm/G_3_gram.fst.txt
    fi

    if [ ! -f data/lm/G_4_gram.fst.txt ]; then
        # It is used for LM rescoring
        ./shared/make_kn_lm.py \
            -ngram-order 4 \
            -text ${lang_dir}/input.txt \
            -lm data/lm/4-gram.arpa
        python3 -m kaldilm \
            --read-symbol-table="data/lang_phone/words.txt" \
            --disambig-symbol='#0' \
            --max-order=4 \
            data/lm/4-gram.arpa >data/lm/G_4_gram.fst.txt
    fi
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    log "Stage 9: Compile HLG"
    ./local/compile_hlg.py --lang-dir data/lang_phone

    # Note If ./local/compile_hlg.py throws OOM,
    # please switch to the following command
    #
    # ./local/compile_hlg_using_openfst.py --lang-dir data/lang_phone

    for vocab_size in ${vocab_sizes[@]}; do
        lang_dir=data/lang_bpe_${vocab_size}
        ./local/compile_hlg.py --lang-dir $lang_dir

        # Note If ./local/compile_hlg.py throws OOM,
        # please switch to the following command
        #
        # ./local/compile_hlg_using_openfst.py --lang-dir $lang_dir
    done
fi

# Compile LG for RNN-T fast_beam_search decoding
if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    log "Stage 10: Compile LG"
    ./local/compile_lg.py --lang-dir data/lang_phone

    for vocab_size in ${vocab_sizes[@]}; do
        lang_dir=data/lang_bpe_${vocab_size}
        ./local/compile_lg.py --lang-dir $lang_dir
    done
fi

if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
    log "Stage 11: Generate LM training data"

    for vocab_size in ${vocab_sizes[@]}; do
        log "Processing vocab_size == ${vocab_size}"
        lang_dir=data/lang_bpe_${vocab_size}
        out_dir=data/lm_training_bpe_${vocab_size}
        mkdir -p $out_dir

        if [ ! -f $out_dir/train.txt ]; then
              tail -n 250000 data/lang_phone/input.txt > $out_dir/train.txt
        fi

        ./local/prepare_lm_training_data.py \
            --bpe-model $lang_dir/bpe.model \
            --lm-data data/lang_phone/input.txt \
            --lm-archive $out_dir/lm_data.pt
    done
fi

if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
    log "Stage 12: Generate LM validation data"

    for vocab_size in ${vocab_sizes[@]}; do
        log "Processing vocab_size == ${vocab_size}"
        out_dir=data/lm_training_bpe_${vocab_size}
        mkdir -p $out_dir

        if [ ! -f $out_dir/valid.txt ]; then
            head -n 14332 data/lang_phone/input.txt > $out_dir/valid.txt
        fi

        lang_dir=data/lang_bpe_${vocab_size}
        ./local/prepare_lm_training_data.py \
            --bpe-model $lang_dir/bpe.model \
            --lm-data $out_dir/valid.txt \
            --lm-archive $out_dir/lm_data-valid.pt
    done
fi

if [ $stage -le 13 ] && [ $stop_stage -ge 13 ]; then
    log "Stage 13: Generate LM test data"
    testsets=(eval2000)

    for testset in ${testsets[@]}; do
        for vocab_size in ${vocab_sizes[@]}; do
            log "Processing vocab_size == ${vocab_size}"
            out_dir=data/lm_training_bpe_${vocab_size}
            mkdir -p $out_dir

            if [ ! -f $out_dir/${testset}.txt ]; then
                gunzip -c data/manifests/${testset}/eval2000_supervisions_all.jsonl.gz \
                    | jq '.text' | sed 's/"//g' > $out_dir/${testset}.txt
            fi

            lang_dir=data/lang_bpe_${vocab_size}
            ./local/prepare_lm_training_data.py \
                --bpe-model $lang_dir/bpe.model \
                --lm-data $out_dir/${testset}.txt \
                --lm-archive $out_dir/lm_data-${testset}.pt
        done
    done
fi

if [ $stage -le 14 ] && [ $stop_stage -ge 14 ]; then
    log "Stage 14: Sort LM training data"
    testsets=(eval2000)
    # Sort LM training data by sentence length in descending order
    # for ease of training.
    #
    # Sentence length equals to the number of BPE tokens
    # in a sentence.

    for vocab_size in ${vocab_sizes[@]}; do
        out_dir=data/lm_training_bpe_${vocab_size}
        mkdir -p $out_dir
        ./local/sort_lm_training_data.py \
            --in-lm-data $out_dir/lm_data.pt \
            --out-lm-data $out_dir/sorted_lm_data.pt \
            --out-statistics $out_dir/statistics.txt
        for testset in ${testsets[@]}; do
            ./local/sort_lm_training_data.py \
                --in-lm-data $out_dir/lm_data-${testset}.pt \
                --out-lm-data $out_dir/sorted_lm_data-${testset}.pt \
                --out-statistics $out_dir/statistics-test-${testset}.txt
        done
    done
fi
