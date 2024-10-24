import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import json
from typing import List, Dict
import logging

from pathlib import Path

import dotenv
import torch

from entropix.config import LLAMA_1B_PARAMS
from entropix.tokenizer import Tokenizer
from entropix.torch_kvcache import KVCache
from entropix.torch_main import build_attn_mask, precompute_freqs_cis
from entropix.torch_model import xfmr
from entropix.torch_sampler import calculate_metrics
from entropix.torch_weights import load_weights

# Setup logging
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(levelname)s - %(message)s",
  handlers=[logging.FileHandler("processing.log"), logging.StreamHandler()],
)

dotenv.load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.inference_mode():
  model_params = LLAMA_1B_PARAMS
  xfmr_weights = load_weights(
    ckpt_dir=Path("weights/1B-Base"), should_compare_outputs=False
  )

  tokenizer = Tokenizer("entropix/tokenizer.model")
  bsz = 1
  kvcache = KVCache.new(
    model_params.n_layers,
    bsz,
    model_params.max_seq_len,
    model_params.n_local_kv_heads,
    model_params.head_dim,
  ).to(DEVICE)


def compute_statistics(prompt, completion):
  stats_arr = []
  with torch.inference_mode():
    prefill_str = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
        {prompt}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    prefill_tokens = tokenizer.encode(
      prefill_str, bos=False, eos=False, allowed_special="all"
    )

    result_str = f"""{completion}<|eot_id|>"""

    joined_str = prefill_str + "\n" + result_str

    joined_tokens = tokenizer.encode(
      joined_str, bos=False, eos=False, allowed_special="all"
    )

    tokens = torch.tensor([joined_tokens], dtype=torch.long).to(DEVICE)
    seqlen = tokens.size(1)

    cur_pos = 0
    freqs_cis = precompute_freqs_cis(
      model_params.head_dim,
      model_params.max_seq_len,
      model_params.rope_theta,
      model_params.use_scaled_rope,
    )
    attn_mask = build_attn_mask(seqlen, cur_pos)
    bsz, seqlen = tokens.shape

    kvcache = KVCache.new(
      model_params.n_layers,
      bsz,
      model_params.max_seq_len,
      model_params.n_local_kv_heads,
      model_params.head_dim,
    ).to(DEVICE)

    logits, kvcache, scores, _ = xfmr(
      xfmr_weights,
      model_params,
      tokens,
      cur_pos,
      freqs_cis[:seqlen],
      kvcache,
      attn_mask=attn_mask,
    )

    for i in range(len(prefill_tokens), len(joined_tokens) - 1):
      ground_truth_token_str = (
        tokenizer.decode([joined_tokens[i + 1]])
        .encode("unicode_escape")
        .decode("utf-8")
      )

      # compute statistics for each token
      mx = calculate_metrics(logits[:, i, :], scores[:, :, :i, :i], i)
      mx_clean = {k: v.item() for k, v in mx.items()}

      mx_results = {}

      # what is the probability of the ground truth token, in the model's output?
      prob = torch.nn.functional.softmax(logits[:, i, :], dim=-1)
      prob_gt = prob[0, joined_tokens[i + 1]].item()
      mx_results["prob"] = prob_gt

      # what is the rank of the ground truth token?
      _, topk = torch.topk(prob, k=10000, dim=-1)
      topk = topk[0].tolist()
      try:
        mx_results["rank"] = topk.index(joined_tokens[i + 1])
      except ValueError:
        mx_results["rank"] = -1

      mx_dict = mx_clean | mx_results
      mx_dict["token"] = ground_truth_token_str
      stats_arr.append(mx_dict)
  return stats_arr


def process_dataset(
  input_path: str, output_path: str, batch_size: int = 100, max_samples: int = None
) -> None:
  """
  Process a dataset through the LLaMA statistics computation.

  Args:
      input_path: Path to input parquet file
      output_path: Path to save the processed results
      batch_size: Number of samples to process before saving
      max_samples: Maximum number of samples to process (None for all)
  """
  # Load the dataset
  logging.info(f"Loading dataset from {input_path}")
  df = pd.read_parquet(input_path)

  if max_samples:
    df = df.head(max_samples)

  # Initialize storage for results
  all_results = []
  current_batch = []

  logging.info(f"Processing {len(df)} samples")

  for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
      # Process each instruction-output pair
      stats = compute_statistics(prompt=row["instruction"], completion=row["output"])

      # Add metadata
      result = {
        "sample_id": idx,
        "instruction": row["instruction"],
        "output": row["output"],
        "statistics": stats,
      }

      current_batch.append(result)

      # Save intermediate results when batch is full
      if len(current_batch) >= batch_size:
        save_batch(current_batch, output_path, len(all_results))
        all_results.extend(current_batch)
        current_batch = []

    except Exception as e:
      logging.error(f"Error processing sample {idx}: {str(e)}")
      continue

  # Save any remaining samples
  if current_batch:
    save_batch(current_batch, output_path, len(all_results))
    all_results.extend(current_batch)

  logging.info(
    f"Processing complete. Processed {len(all_results)} samples successfully"
  )

def save_batch(batch: List[Dict], output_path: str, start_idx: int) -> None:
  """Save a batch of results to a JSON file."""
  batch_path = Path(output_path) / f"batch_{start_idx:06d}.json"
  with open(batch_path, "w") as f:
    json.dump(batch, f)
  logging.info(f"Saved batch to {batch_path}")


def main():
  # Create output directory if it doesn't exist
  output_dir = Path("processed_results")
  output_dir.mkdir(exist_ok=True)

  # Process the dataset
  process_dataset(
    input_path="hf://datasets/tatsu-lab/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
    output_path=str(output_dir),
    batch_size=100,
    max_samples=None,  # Set to a number for testing
  )


if __name__ == "__main__":
  main()
