"""
Calculate UTMOSv2 score with automatic Mean Opinion Score (MOS) prediction system
"""

import argparse
import logging
import os
import utmosv2

import numpy as np

logging.basicConfig(level=logging.INFO)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wav-path", type=str, help="path of the evaluated speech directory"
    )
    parser.add_argument(
        "--utmos-model-path",
        type=str,
        default="TTS_eval_models/utmosv2/fold0_s42_best_model.pth",
        help="path of the UTMOS model",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    model = utmosv2.create_model(pretrained=True, checkpoint_path=args.utmos_model_path)
    score_dict_list = model.predict(input_dir=args.wav_path)
    score_list = [score_dict["predicted_mos"] for score_dict in score_dict_list]
    logging.info(f"UTMOS score: {np.mean(score_list):.2f}")
