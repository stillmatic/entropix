import torch
import torch.nn.functional as F
from typing import Tuple, Dict

# Device selection, tree is like first apple silicion, then cuda, fallback is cpu.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

def calculate_varentropy_logsoftmax(logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = F.log_softmax(logits, dim=axis)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2  # Convert to base-2
    varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=axis)
    return entropy, varentropy

def multinomial_sample_one(probs_sort: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Samples one token from a multinomial distribution with sorted probabilities."""
    # Use torch.rand instead of Exponential distribution
    q = torch.rand(probs_sort.shape, generator=generator, device=probs_sort.device)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(torch.int32)

def _sample(logits: torch.Tensor, temperature=0.666, top_p=0.90, top_k=27, min_p: float = 0.0, 
            repetition_penalty: float = 1.0,
            prev_tokens: torch.Tensor = None,
            generator: torch.Generator = None) -> torch.Tensor:
    bsz = logits.shape[0]
    logit = logits[:, -1]

    if repetition_penalty != 1.0 and prev_tokens is not None:
        score = torch.gather(logit, 1, prev_tokens)
        # if score < 0 then repetition penalty has to be multiplied
        # if score > 0 then repetition penalty has to be divided
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        logit.scatter_(1, prev_tokens, score)

    probs = F.softmax(logit / temperature, dim=-1)

    # Apply min_p sampling
    if min_p > 0.0:
        p_max = torch.max(probs, dim=-1, keepdim=True).values
        indices_to_remove = probs < (min_p * p_max)
        logit = torch.where(indices_to_remove, torch.full_like(logit, float('-inf')), logit)

    # Apply top-k sampling
    top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]))
    probs_sort = torch.flip(top_k_probs, dims=[-1])
    probs_idx = torch.flip(top_k_indices, dims=[-1])
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # Apply top-p sampling
    mask = torch.where(probs_sum - probs_sort > top_p, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
    probs_sort = probs_sort * (1 - mask)
    probs_sort = probs_sort / torch.sum(probs_sort, dim=-1, keepdim=True)
    next_token = multinomial_sample_one(probs_sort, generator)
    # Convert next_token to int64 before using it in gather
    next_token_g = torch.gather(probs_idx, -1, next_token.reshape(bsz, 1).to(torch.int64))
    return next_token_g.to(torch.int32)

def calculate_metrics(logits: torch.Tensor, attention_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)

    # NB chua: filter to non-zero values because future values are always zero (causal mask)
    # another implementation would be to pass in or calculate the number of indices at play
    # this is _probably_ fine though.
    attention_scores = torch.where(attention_scores != 0.0, attention_scores, torch.full_like(attention_scores, float('-inf')))

    attention_probs = F.softmax(attention_scores, dim=-1)
    attn_entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, 1e-10, 1.0)), dim=-1)
    attn_varentropy = torch.var(attn_entropy, dim=1)
    
    # Add a small epsilon to avoid NaN when all values are the same
    attn_varentropy = torch.where(torch.isnan(attn_varentropy), torch.zeros_like(attn_varentropy), attn_varentropy)
    mean_attention = torch.mean(attention_probs, dim=1)
    # agreement is a measure of how much each attention layer agrees with the mean attention
    agreement = torch.mean(torch.abs(attention_probs - mean_attention.unsqueeze(1)), dim=(1, 2))
    non_inf_attn_scores = torch.where(attention_scores != float('-inf'), attention_scores, torch.full_like(attention_scores, torch.nan))
    interaction_strength = torch.nanmean(torch.abs(non_inf_attn_scores), dim=(1, 2, 3))

    metrics =  {
        "logits_entropy": torch.mean(entropy),
        "logits_varentropy": torch.mean(varentropy),
        "attn_entropy": torch.mean(attn_entropy),
        "attn_varentropy": torch.mean(attn_varentropy),
        "agreement": torch.mean(agreement),
        "interaction_strength": interaction_strength
    }
    return metrics

def adaptive_sample(logits: torch.Tensor, metrics: Dict[str, torch.Tensor],
                    gen_tokens: torch.Tensor, n_samples: int,
                    base_temp: float = 0.666, base_top_p: float = 0.90, base_top_k: int = 40, base_min_p: float = 0.01,
                    generator: torch.Generator = None) -> torch.Tensor:
    logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
    attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]

    temperature = base_temp * (1 + 0.3 * logits_uncertainty + 0.2 * attn_uncertainty - 2 * metrics["agreement"])
    top_p = torch.clamp(base_top_p * (1 + 0.1 * metrics["attn_varentropy"]), 0.1, 1.0)
    top_k = int(torch.clamp(
        torch.round(torch.tensor(base_top_k) * (1 + 0.3 * metrics["interaction_strength"].item() - 2 * metrics["agreement"].item())),
        min=1,
        max=100
    ).item())
    min_p = torch.clamp(base_min_p * (2 - 0.05 * logits_uncertainty), 0.01, 0.5)
    # print(" ", "min_p", min_p.item(), "temperature", temperature.item(), "top_p", top_p.item(), "top_k", top_k)

    samples = []
    for _ in range(n_samples):
        sample = _sample(logits, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p, generator=generator)
        samples.append(sample)

    def score_sample(sample):
        # Flatten the sample tensor and convert to long (int64)
        sample_flat = sample.flatten().to(torch.long)
        
        # Create one-hot encoding
        one_hot = F.one_hot(sample_flat, logits.shape[-1])
        
        # Reshape log_softmax output to match one_hot
        log_probs = F.log_softmax(logits, dim=-1).view(-1, logits.shape[-1])
        
        # Calculate log probability
        log_prob = torch.sum(log_probs * one_hot)
        
        confidence_score = (
            (1 - metrics["logits_entropy"]) * 0.1 +
            (1 - metrics["attn_entropy"]) * 0.2 +
            (1 - metrics["logits_varentropy"]) * 0.3 +
            (1 - metrics["attn_varentropy"]) * 0.4 +
            metrics["agreement"] * 0.5 +
            metrics["interaction_strength"] * 0.6
        )
        return log_prob + confidence_score

    sample_scores = torch.stack([score_sample(sample) for sample in samples])
    best_sample_idx = torch.argmax(sample_scores)
    return samples[best_sample_idx]


COT_TOKENS = [2564, 14524, 81122, 11748, 12174, 14524, 2319]

def sample(gen_tokens: torch.Tensor, logits: torch.Tensor, attention_scores: torch.Tensor,
           temperature=0.666, top_p=0.90, top_k=27, min_p: float = 0.0, 
           generator: torch.Generator = torch.Generator(device=device).manual_seed(1234)) -> torch.Tensor:
    metrics = calculate_metrics(logits, attention_scores)
    # print({k: v.item() for k, v in metrics.items()})
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if ent < 0.1 and vent < 0.3:
        return torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif ent > 3.0 and vent < 1.3:
        print("Asking clarifying question\n")
        # Insert a clarifying question token if not already present
        if not torch.isin(gen_tokens[:,-1], torch.tensor([2564], device=device)).any():
            rand_cot_token = torch.tensor([COT_TOKENS[torch.randint(0, len(COT_TOKENS), (1,))]], device=device)
            return rand_cot_token # Assuming 2564 is our "ask clarifying question" token
        else:
            # If we've just asked a question, sample with slightly higher temperature
            temp_adj = 1.3 + 0.2 * attn_ent  # Increase temperature based on attention entropy
            return _sample(logits, temperature=min(1.5, temperature * temp_adj), top_p=top_p, top_k=top_k, min_p=min_p, generator=generator)

    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif ent < 5.0 and vent > 5.0:
        # print(" forking\n")
        temp_adj = 1.2 + 0.03 * interaction_strength  # Increase temperature based on interaction strength
        top_k_adj = max(5, int(top_k * (1 + 0.5 * (1 - agreement))))  # Increase top_k when agreement is low
        return _sample(logits, temperature=min(1.5, temperature * temp_adj), top_p=top_p, top_k=top_k_adj, min_p=min_p, generator=generator)

    # High Entropy, High Varentropy: "resampling in the mist"
    elif ent > 5.0 and vent > 5.0:
        # print(" resampling\n")
        # Use high temperature and adjusted top_p based on attention metrics
        temp_adj = 2.0 + 0.5 * attn_vent  # Increase temperature based on attention varentropy
        top_p_adj = max(0.5, top_p - 0.2 * attn_ent)  # Decrease top_p when attention entropy is high
        return _sample(logits, temperature=max(2.0, temperature * temp_adj), top_p=top_p_adj, top_k=top_k, min_p=min_p, generator=generator)

    # Middle ground: use adaptive sampling
    else:
        return adaptive_sample(
            logits,
            metrics,
            gen_tokens,
            n_samples=5,
            base_temp=temperature,
            base_top_p=top_p,
            base_top_k=top_k,
            generator=generator
        )