# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Resolve one-based audio-codebook sampling overrides across stacked frames."""

import math


def validate_top_p(top_p):
    """Validate the audio nucleus threshold; 1 disables nucleus filtering."""
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise ValueError('top_p must be finite and in (0, 1]')


def filter_top_p(logits, top_p):
    """Filter temperature-scaled, optionally top-k-filtered logits on their last axis.

    Keep the smallest descending-probability prefix reaching top_p, including the
    boundary token and at least one token. Probabilities are relative to the input
    support, so combining top-k and top-p applies the nucleus within top-k.
    """
    validate_top_p(top_p)
    if top_p == 1:
        return logits
    sorted_logits, indices = logits.sort(dim=-1, descending=True)
    cumulative = sorted_logits.float().softmax(dim=-1).cumsum(dim=-1)
    remove = cumulative >= top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    return logits.masked_fill(remove.scatter(-1, indices, remove), float('-inf'))


def resolve_codebook_temperature(temperature, overrides, num_codebooks, stacking_factor):
    """Expand one-based temperature overrides to all stacked frames; zero means greedy."""
    return _resolve_codebook_float(temperature, overrides, num_codebooks, stacking_factor, 'temperature')


def resolve_codebook_cfg_scale(cfg_scale, overrides, num_codebooks, stacking_factor):
    """Expand one-based audio CFG scales to every stacked frame without changing phoneme CFG."""
    return _resolve_codebook_float(cfg_scale, overrides, num_codebooks, stacking_factor, 'cfg_scale')


def resolve_codebook_topk(topk, overrides, num_codebooks, stacking_factor, vocabulary_size):
    """Return codebook-major channel top-k values, or None for the unchanged scalar path.

    ``overrides`` maps one-based codebook numbers to top-k. Each override applies to
    every stacked frame of that codebook; omitted codebooks retain ``topk``.
    """
    if overrides is None or overrides == {}:
        return None
    if not isinstance(overrides, dict):
        raise ValueError('codebook_topk must be a mapping of one-based codebook numbers to top-k')
    if isinstance(topk, bool) or not isinstance(topk, int) or not 0 <= topk <= vocabulary_size:
        raise ValueError(f'topk must be in [0, {vocabulary_size}] (0 disables top-k)')
    resolved = [topk] * (num_codebooks * stacking_factor)
    seen = set()
    for key, value in overrides.items():
        if isinstance(key, bool) or not isinstance(key, (str, int)) or not str(key).isdigit():
            raise ValueError(f'Invalid one-based codebook number: {key!r}')
        codebook = int(key)
        if not 1 <= codebook <= num_codebooks or codebook in seen:
            raise ValueError(f'Codebook numbers must be unique and in [1, {num_codebooks}]: {key!r}')
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= vocabulary_size:
            raise ValueError(f'Codebook {codebook} top-k must be an integer in [1, {vocabulary_size}]')
        seen.add(codebook)
        start = (codebook - 1) * stacking_factor
        resolved[start : start + stacking_factor] = [value] * stacking_factor
    return tuple(resolved)


def _resolve_codebook_float(default, overrides, num_codebooks, stacking_factor, label):
    if overrides is None or overrides == {}:
        return None
    if not isinstance(overrides, dict):
        raise ValueError(f'codebook_{label} must be a mapping of one-based codebooks to values')
    resolved = [float(default)] * (num_codebooks * stacking_factor)
    seen = set()
    for key, value in overrides.items():
        if isinstance(key, bool) or not isinstance(key, (str, int)) or not str(key).isdigit():
            raise ValueError(f'Invalid one-based codebook number: {key!r}')
        codebook = int(key)
        if not 1 <= codebook <= num_codebooks or codebook in seen:
            raise ValueError(f'Codebook numbers must be unique and in [1, {num_codebooks}]: {key!r}')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Codebook {codebook} {label} must be finite and non-negative')
        seen.add(codebook)
        start = (codebook - 1) * stacking_factor
        resolved[start : start + stacking_factor] = [float(value)] * stacking_factor
    if not math.isfinite(default) or default < 0:
        raise ValueError(f'Base {label} must be finite and non-negative')
    return tuple(resolved)
