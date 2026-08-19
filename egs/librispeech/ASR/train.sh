#!/usr/bin/env bash

ICEFALL=/star-kw/kangwei/worktree/zapformer_preview
export PYTHONPATH=$ICEFALL:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export NCCL_TOPO_FILE=/etc/topo/topo_h800_hpc_bcc.xml
export NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8

stage=0
stop_stage=10

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. shared/parse_options.sh || exit 1

if [ $stage -le 100 ] && [ $stop_stage -ge 100 ]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

    python zapformer/train_bpe.py \
      --world-size 8 \
      --master-port 12396 \
      --num-real-epochs 20 \
      --min-copies 1 \
      --max-copies 2 \
      --start-epoch 1 \
      --use-fp16 1 \
      --batches-per-epoch 9000 \
      --train-manifests data/fbank/wenetspeech_cuts_L_fixed_1w.jsonl.gz \
                        data/fbank/gigaspeech_cuts_XL.jsonl.gz \
      --valid-manifests data/fbank/wenetspeech_cuts_DEV.jsonl.gz \
                        data/fbank/gigaspeech_cuts_DEV.jsonl.gz \
      --exp-dir exp/zapformer/base-dim-64 \
      --bpe-model zh-en-8776 \
      --use-ctc 1 \
      --use-transducer 1 \
      --base-dim 64 \
      --ctc-loss-scale 0.2 \
      --base-lr 0.023 \
      --num-workers 4 \
      --max-duration 1000
fi


if [ $stage -le 102 ] && [ $stop_stage -ge 102 ]; then
    (
    export CUDA_VISIBLE_DEVICES="4"
    for avg in 2; do
      python zapformer/decode_wds_simple.py \
        --epoch 18 \
        --avg $avg \
        --test-sets reference,data/tars/reference.lst \
                  kespeech,data/tars/kespeech_test.lst \
                  evaluation,data/tars/evaluation.lst \
        --exp-dir exp/zapformer/base-dim-64 \
        --bpe-model data/zh-10103.vocab \
        --use-ctc 1 \
        --use-transducer 1 \
        --base-dim 64 \
        --max-duration 400 \
        --decoding-method rnnt-greedy-search
    done
    ) &

    (
    export CUDA_VISIBLE_DEVICES="5"
    for avg in 2; do
      python zapformer/decode_wds_simple.py \
        --epoch 18 \
        --avg $avg \
        --test-sets reference,data/tars/reference.lst \
                  kespeech,data/tars/kespeech_test.lst \
                  evaluation,data/tars/evaluation.lst \
        --exp-dir exp/zapformer/base-dim-64 \
        --bpe-model data/zh-10103.vocab \
        --use-ctc 1 \
        --use-transducer 1 \
        --base-dim 64 \
        --max-duration 400 \
        --decoding-method ctc-greedy-search
    done
    ) &
    wait
fi

if [ $stage -le 106 ] && [ $stop_stage -ge 106 ]; then
    for item in 0,0.02,0.3 0,0.03,0.3 1,0.01,0.2 1,0.01,0.4 2,0.01,0.25 2,0.01,0.35 3,0.01,0.3 3,0.015,0.3; do
      (
        export CUDA_VISIBLE_DEVICES=$(echo $item | cut -d, -f1)
        lm_scale=$(echo $item | cut -d, -f3)
        ngram_scale=$(echo $item | cut -d, -f2)
        python zapformer/decode.py \
          --epoch 15 \
          --avg 2 \
          --test-sets reference,data/tars/reference.lst \
                      kespeech,data/tars/kespeech_test.lst \
          --exp-dir exp/zapformer/base-dim-64 \
          --bpe-model data/zh-10103.vocab \
          --use-ctc 1 \
          --use-transducer 1 \
          --base-dim 64 \
          --use-shallow-fusion 1 \
          --lm-vocab-size 10103 \
          --backoff-id 10103 \
          --lm-epoch 99 \
          --lm-avg 2 \
          --lm-exp-dir /star-ssd/kangwei/worktree/zapformer_3346/egs/librispeech/ASR/data/ngram/exp-lr5e-4-dtiny \
          --lm-scale $lm_scale \
          --ngram-lm-scale $ngram_scale \
          --lang-dir /star-ssd/kangwei/worktree/zapformer_3346/egs/librispeech/ASR/data/ngram \
          --max-duration 400 \
          --decoding-method modified_beam_search_LODR
      ) &
    done
    wait
fi
