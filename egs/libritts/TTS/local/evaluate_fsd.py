"""
Calculate Frechet Speech Distance betweeen two speech directories.

# Download wav2vec 2.0 model
huggingface-cli download facebook/wav2vec2-base --local-dir model/huggingface/wav2vec2_base

"""
import argparse
import logging
import os
from multiprocessing.dummy import Pool as ThreadPool

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy import linalg
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

logging.basicConfig(level=logging.INFO)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--real-path", type=str, help="path of the real speech directory"
    )
    parser.add_argument(
        "--eval-path", type=str, help="path of the evaluated speech directory"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="TTS_eval_models/wav2vec2_base",
        help="path of the wav2vec 2.0 model directory",
    )
    parser.add_argument(
        "--real-embds-path",
        type=str,
        default=None,
        help="path of the real embedding directory",
    )
    parser.add_argument(
        "--eval-embds-path",
        type=str,
        default=None,
        help="path of the evaluated embedding directory",
    )
    return parser


class FrechetSpeechDistance:
    def __init__(
        self,
        model_path="TTS_eval_models/wav2vec2_base",
        pca_dim=128,
        speech_load_worker=8,
    ):
        """
        Initialize FSD
        """
        self.sample_rate = 16000
        self.channels = 1
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logging.info("[Frechet Speech Distance] Using device: {}".format(self.device))
        self.speech_load_worker = speech_load_worker
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = Wav2Vec2Model.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.pca_dim = pca_dim

    def get_embeddings(self, dir, dtype="float32"):
        """
        Get embeddings
        """

        def _load_speech_task(fname, sample_rate):

            wav_data, sr = sf.read(fname, dtype=dtype)
            if sr != sample_rate:
                wav_data = librosa.resample(
                    wav_data, orig_sr=sr, target_sr=self.sample_rate
                )
            wav_data = torch.from_numpy(wav_data)

            return wav_data

        embd_lst = []
        for fname in tqdm(os.listdir(dir)):
            if os.path.splitext(fname)[-1] not in [
                ".wav",
                ".flac",
                ".mp3",
                ".ogg",
                ".m4a",
                ".opus",
            ]:
                logging.warning(f"skipping {fname}")
                continue
            speech = _load_speech_task(os.path.join(dir, fname), self.sample_rate)
            input_features = self.feature_extractor(
                speech, sampling_rate=self.sample_rate, return_tensors="pt"
            ).input_values.to(self.device)
            with torch.no_grad():
                embd = self.model(input_features).last_hidden_state.mean(1)

            if embd.device != torch.device("cpu"):
                embd = embd.cpu()

            if torch.is_tensor(embd):
                embd = embd.detach().numpy()

                embd_lst.append(embd)
        return np.concatenate(embd_lst, axis=0)

    def calculate_embd_statistics(self, embd_lst):
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2).astype(complex), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            logging.info(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm(
                (sigma1 + offset).dot(sigma2 + offset).astype(complex)
            )

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def score(
        self,
        real_path,
        eval_path,
        real_embds_path=None,
        eval_embds_path=None,
        dtype="float32",
    ):
        """
        Computes the Frechet Speech Distance (FSD) between two directories of speech files.

        Parameters:
        - real_path (str): Path to the directory containing real speech files.
        - eval_path (str): Path to the directory containing evaluation speech files.
        - real_embds_path (str, optional): Path to save/load real speech embeddings (e.g., /folder/bkg_embs.npy). If None, embeddings won't be saved.
        - eval_embds_path (str, optional): Path to save/load evaluation speech embeddings (e.g., /folder/test_embs.npy). If None, embeddings won't be saved.
        - dtype (str, optional): Data type for loading speech. Default is "float32".

        Returns:
        - float: The Frechet Speech Distance (FSD) score between the two directories of speech files.
        """
        # Load or compute real embeddings
        if real_embds_path is not None and os.path.exists(real_embds_path):
            logging.info(
                f"[Frechet Speech Distance] Loading embeddings from {real_embds_path}..."
            )
            embds_real = np.load(real_embds_path)
        else:
            embds_real = self.get_embeddings(real_path, dtype=dtype)
            if real_embds_path:
                os.makedirs(os.path.dirname(real_embds_path), exist_ok=True)
                np.save(real_embds_path, embds_real)

        # Load or compute eval embeddings
        if eval_embds_path is not None and os.path.exists(eval_embds_path):
            logging.info(
                f"[Frechet Speech Distance] Loading embeddings from {eval_embds_path}..."
            )
            embds_eval = np.load(eval_embds_path)
        else:
            embds_eval = self.get_embeddings(eval_path, dtype=dtype)
            if eval_embds_path:
                os.makedirs(os.path.dirname(eval_embds_path), exist_ok=True)
                np.save(eval_embds_path, embds_eval)

        # Check if embeddings are empty
        if len(embds_real) == 0:
            logging.info("[Frechet Speech Distance] real set dir is empty, exiting...")
            return -1
        if len(embds_eval) == 0:
            logging.info("[Frechet Speech Distance] eval set dir is empty, exiting...")
            return -1

        # Compute statistics and FSD score
        mu_real, sigma_real = self.calculate_embd_statistics(embds_real)
        mu_eval, sigma_eval = self.calculate_embd_statistics(embds_eval)

        fsd_score = self.calculate_frechet_distance(
            mu_real, sigma_real, mu_eval, sigma_eval
        )

        return fsd_score


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    FSD = FrechetSpeechDistance(model_path=args.model_path)
    score = FSD.score(
        args.real_path, args.eval_path, args.real_embds_path, args.eval_embds_path
    )
    logging.info(f"FSD score: {score:.2f}")
