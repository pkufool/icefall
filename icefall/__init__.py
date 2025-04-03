# isort:skip_file

try:
    import k2
except Exception as ex:
    print("k2 is not installed correctly. Some features might not work.")
    print(
        "If you encounter errors later, please refer to https://k2-fsa.org/get-started/k2/"
    )

import sys

from . import checkpoint, decode, dist, env, utils

from .byte_utils import (
    byte_decode,
    byte_encode,
    smart_byte_decode,
)

from .checkpoint import (
    average_checkpoints,
    find_checkpoints,
    load_checkpoint,
    remove_checkpoints,
    save_checkpoint,
    save_checkpoint_with_global_batch_idx,
)

from .context_graph import ContextGraph, ContextState

if "k2" in sys.modules:
    from .decode import (
        get_lattice,
        nbest_decoding,
        nbest_oracle,
        one_best_decoding,
        rescore_with_attention_decoder,
        rescore_with_n_best_list,
        rescore_with_whole_lattice,
    )

from .dist import (
    cleanup_dist,
    setup_dist,
)

from .env import (
    get_env_info,
    get_git_branch_name,
    get_git_date,
    get_git_sha1,
)

from .utils import (
    AttributeDict,
    MetricsTracker,
    encode_supervisions,
    get_executor,
    is_cjk,
    is_jit_tracing,
    is_module_available,
    l1_norm,
    l2_norm,
    linf_norm,
    load_alignments,
    make_pad_mask,
    measure_gradient_norms,
    measure_weight_norms,
    optim_step_and_measure_param_change,
    save_alignments,
    setup_logger,
    store_transcripts,
    str2bool,
    subsequent_chunk_mask,
    tokenize_by_CJK_char,
    tokenize_by_ja_char,
    write_error_stats,
)

if "k2" in sys.modules:
    from .utils import (
        add_eos,
        add_sos,
        concat,
        get_alignments,
        get_texts,
    )


from .ngram_lm import NgramLm, NgramLmStateCost

from .lm_wrapper import LmScorer
