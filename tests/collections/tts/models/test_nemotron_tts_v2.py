# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nemo.collections.tts.models.nemotron_tts import NemotronTTSModel
from tests.collections.tts.models.test_easy_magpietts import (
    _codec_restore_uses_random_initialized_audio_codec,
    _seed_everything,
    _toy_batch,
    tiny_easy_magpie_cfg,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "prediction_order", [None, "codebook_major", "frame_major"], ids=["default", "codebook", "frame"]
)
def test_acoustic_codes_predictor_process_batch_with_frame_stacking(prediction_order):
    model = _make_model(prediction_order)
    output = model.process_batch(**_toy_batch(model), mode="val", training_mode=model.training_modes[0])
    predictor = model.acoustic_codes_predictor
    predicted = predictor.predict_codes(torch.randn(2, 1, model.cfg.hidden_dim), temperature=0.0)

    expected_order = (
        list(range(0, 16, 2)) + list(range(1, 16, 2)) if prediction_order == "frame_major" else list(range(16))
    )
    assert predictor.prediction_order == (prediction_order or "codebook_major")
    assert [i for block in predictor.blocks for i in block.codebook_indices] == expected_order
    assert predicted.shape == (2, 1, 2 * model.num_audio_codebooks)
    assert output.audio_codes_target.size(1) == 2 * model.num_audio_codebooks
    assert torch.isfinite(output.acoustic_codes_predictor_loss)
    assert torch.isfinite(output.loss)


def test_acoustic_codes_predictor_frame_major_rejects_grouped_schedule_from_model_config():
    with pytest.raises(ValueError, match="requires .* to contain only ones"):
        _make_model("frame_major", schedule=[4, 12])


def _make_model(prediction_order, schedule=None):
    _seed_everything()
    cfg = tiny_easy_magpie_cfg(
        {
            "use_codec_latent_audio_embedding": True,
            "use_bpe_char_tokenizer": False,
            "frame_stacking_factor": 2,
            "acoustic_codes_predictor_schedule": (
                schedule if schedule is not None else ([1] * 16 if prediction_order == "frame_major" else [4, 12])
            ),
        }
    )
    # Use a local character tokenizer: prediction-order tests do not need Hugging Face assets.
    cfg.text_tokenizers = {
        "nemotron_bpe": {
            "_target_": "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.EnglishCharsTokenizer",
        }
    }
    if prediction_order is not None:
        cfg.acoustic_codes_predictor_order = prediction_order
    with _codec_restore_uses_random_initialized_audio_codec():
        model = NemotronTTSModel(cfg)
    model.eval()
    return model
