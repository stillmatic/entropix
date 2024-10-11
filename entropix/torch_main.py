import json
import math
from functools import partial
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
import tyro
from rich import print
from rich.console import Console

from entropix.config import LLAMA_1B_PARAMS
from entropix.prompts import bp1, bp4, prompt, default_prompt
from entropix.tokenizer import Tokenizer
from entropix.torch_kvcache import KVCache
from entropix.torch_model import xfmr
from entropix.torch_sampler import calculate_metrics, sample
from entropix.torch_weights import LayerWeights, XfmrWeights, load_weights

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# prompt = bp4
prompt = default_prompt

# Device selection, tree is like first apple silicion, then cuda, fallback is cpu.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

torch.set_float32_matmul_precision('high')

def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    SCALE_FACTOR = 8.0
    LOW_FREQ_FACTOR = 1.0
    HIGH_FREQ_FACTOR = 4.0
    OLD_CONTEXT_LEN = 8192  # original llama3 length

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq: torch.Tensor) -> torch.Tensor:
        wavelen = 2 * torch.pi / freq

        # Calculate smooth factor
        smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
        smooth = torch.clamp(smooth, 0.0, 1.0)  # Ensure smooth is between 0 and 1

        # Calculate scaled frequency
        scaled = (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        # Apply conditional scaling
        scaled = torch.where(
            wavelen < high_freq_wavelen,
            freq,  # No scaling
            torch.where(
                wavelen > low_freq_wavelen,
                freq / SCALE_FACTOR,  # Apply scaling factor
                scaled  # Apply smooth scaling
            )
        )
        return scaled

    scaled_freqs = torch.vmap(scale_freq)(freqs)
    
    return scaled_freqs

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype, device=device)[: (dim // 2)] / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)

    t = torch.arange(end, dtype=dtype, device=device).unsqueeze(1)  # Shape: (end, 1)
    freqs = freqs.unsqueeze(0)  # Shape: (1, dim//2)
    freqs = t * freqs  # Broadcasting to shape: (end, dim//2)
    return torch.exp(1j * freqs)



def build_attn_mask(seqlen: int, start_pos: int) -> torch.Tensor:
  mask = None
  if seqlen > 1:
      mask = torch.full((seqlen, seqlen), float("-inf"))
      mask = torch.triu(mask, diagonal=1)
      mask = torch.hstack([torch.zeros((seqlen, start_pos)), mask]).to(torch.float32).to(device)
  return mask



def main():
  import pandas as pd

  console = Console()
  with torch.inference_mode():
    model_params = LLAMA_1B_PARAMS
    xfmr_weights = load_weights(ckpt_dir=Path("weights/1B-Instruct"), should_compare_outputs=True)

    tokenizer = Tokenizer('entropix/tokenizer.model')
    raw_tokens1 = tokenizer.encode(prompt,  bos=False, eos=False, allowed_special='all')
    #this is not used in this script, but can be used to generate base_raw_tokens1
    base_raw_tokens1 = tokenizer.encode(bp1, bos=True, eos=False, allowed_special='all')
    # torch.manual_seed(1234)


    def generate(xfmr_weights, model_params, tokens):
      gen_tokens = None
      cur_pos = 0
      tokens = torch.tensor([tokens], dtype=torch.long).to(device)
      bsz, seqlen = tokens.shape
      attn_mask = build_attn_mask(seqlen, cur_pos)
      freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
      kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim).to(DEVICE)
      logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
      next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
      gen_tokens = next_token
      console.print(tokenizer.decode([next_token.item()]), end='')
      cur_pos = seqlen
      stop = torch.tensor([128001, 128008, 128009], device=device, dtype=torch.int32)
      stat_list = []
      num_recent_deletes = 0
      should_noise = False
      while cur_pos < 512:
        cur_pos += 1
        # print("cur_pos", cur_pos)
        logits, kvcache, scores, stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
        if should_noise:
           logits = logits + torch.randn_like(logits) * 0.1 
        metrics = calculate_metrics(logits, scores)
        metrics_clean = {k: v.item() for k, v in metrics.items()}

        # print(metrics_clean)

        ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
        del metrics

        # basic weighting to prevent backspacing too much
        threshold = 5.0 + 2 * num_recent_deletes
        if ent > threshold and vent > threshold and cur_pos > seqlen + 4:
        #    backspace and pop the last token
            num_recent_deletes += 1
            # reset to the position before the last token, regenerate the token
            cur_pos -= 2
            next_token = gen_tokens[:, -2].unsqueeze(0)
            gen_tokens = gen_tokens[:, :-1]
            console.print("⌫", end='', style="red")
            should_noise = True
            continue
        else:
            num_recent_deletes = max(0, num_recent_deletes - 0.5)
            should_noise = False

        temperature = 0.7 + (0.5 * num_recent_deletes)
        next_token = sample(gen_tokens, logits, scores, temperature=temperature)

        token_str = tokenizer.decode(next_token.tolist()[0])
        stat_list.append({'token': token_str, **metrics_clean})
        gen_tokens = torch.cat((gen_tokens, next_token), dim=1)

        style = ""
        if metrics_clean["logits_entropy"] > 3:
          style = "bold"
        elif metrics_clean["logits_varentropy"] > 15:
           style += "blue"
        console.print(token_str, end='', style=style)
        if torch.isin(next_token, stop).any():
          break
    
      df = pd.DataFrame(stat_list)
      df.to_csv('stats.csv', index=False)

    #   print tokens
      raw_tokens = gen_tokens[0].cpu().numpy().tolist()
      console.print("\n---", style="bold")
      console.print(tokenizer.decode(raw_tokens), style="green")

    print(prompt)
    generate(xfmr_weights, model_params, raw_tokens1)

if __name__ == '__main__': 
  tyro.cli(main)