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
      --batches-per-epoch 10000 \
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
      --max-duration 900
fi

# base-dim-64 CTC greedy search decoding
if [ $stage -le 200 ] && [ $stop_stage -ge 200 ]; then
    export CUDA_VISIBLE_DEVICES="4"

    python zapformer/decode_bpe.py \
      --epoch 15 \
      --avg 2 \
      --use-averaged-model true \
      --exp-dir exp/zapformer/base-dim-64 \
      --bpe-model zh-en-8776 \
      --decoding-method ctc-greedy-search \
      --base-dim 64 \
      --max-duration 300 \
      --num-workers 2 \
      --test-manifest gigaspeech_TEST,data/fbank/gigaspeech_cuts_TEST.jsonl.gz \
      --test-manifest wenetspeech_TEST_NET,data/fbank/wenetspeech_cuts_TEST_NET.jsonl.gz \
      --test-manifest wenetspeech_TEST_MEETING,data/fbank/wenetspeech_cuts_TEST_MEETING.jsonl.gz
fi

# base-dim-64 RNNT greedy search decoding
if [ $stage -le 201 ] && [ $stop_stage -ge 201 ]; then
    export CUDA_VISIBLE_DEVICES="5"

    python zapformer/decode_bpe.py \
      --epoch 15 \
      --avg 2 \
      --use-averaged-model true \
      --exp-dir exp/zapformer/base-dim-64 \
      --bpe-model zh-en-8776 \
      --decoding-method rnnt-greedy-search \
      --base-dim 64 \
      --max-duration 300 \
      --num-workers 2 \
      --test-manifest gigaspeech_TEST,data/fbank/gigaspeech_cuts_TEST.jsonl.gz \
      --test-manifest wenetspeech_TEST_NET,data/fbank/wenetspeech_cuts_TEST_NET.jsonl.gz \
      --test-manifest wenetspeech_TEST_MEETING,data/fbank/wenetspeech_cuts_TEST_MEETING.jsonl.gz
fi

