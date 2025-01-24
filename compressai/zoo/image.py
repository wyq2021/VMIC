# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from torch.hub import load_state_dict_from_url

from compressai.models import (
    VMIC,
)

from .pretrained import load_pretrained

__all__ = [
    "vmic",
]

model_architectures = {
    "vmic":VMIC,
}


own_model = {
    "vmic": {
        "mse": {
            1: f"/HDD/1/checkpoint_best_loss-d7c2db72.pth.tar",
            2: f"/HDD/2/checkpoint_best_loss-e5528e0f.pth.tar",
            3: f"/HDD/3/checkpoint_best_loss-de2b0784.pth.tar",
            4: f"/HDD/4/checkpoint_best_loss-fc7e99b1.pth.tar",
            5: f"/HDD/5/checkpoint_best_loss-9876ba3b.pth.tar",
            6: f"/HDD/6/checkpoint_best_loss-e46b47ff.pth.tar",
            7: f"/HDD/7/checkpoint_best_loss-43099225.pth.tar",
        },
        "ms-ssim": {
            1: f"/HDD/1/ms-ssim-checkpoint_best_loss-a926c92d.pth.tar",
            2: f"/HDD/2/ms-ssim-checkpoint_best_loss-d7d027da.pth.tar",
            5: f"/HDD/5/ms-ssim-checkpoint_best_loss-da151765.pth.tar",
            6: f"/HDD/6/ms-ssim-checkpoint_best_loss-bf7bee11.pth.tar",
            7: f"/HDD/7/ms-ssim-checkpoint_best_loss-315aee46.pth.tar",
        },
    },
}


cfgs = {
    "vmic": {
        1: (192, 320),
        2: (192, 320),
        3: (192, 320),
        4: (192, 320),
        5: (192, 320),
        6: (192, 320),
    },
}


def _load_model(
    architecture, metric, quality, pretrained=False, progress=True, **kwargs
):
    if architecture not in model_architectures:
        raise ValueError(f'Invalid architecture name "{architecture}"')

    if quality not in cfgs[architecture]:
        raise ValueError(f'Invalid quality value "{quality}"')

    if pretrained:
        
        if architecture in own_model:
            model_path = own_model[architecture][metric][quality]
            model = torch.load(model_path)
            state_dict = load_pretrained(model)
            model = model_architectures[architecture].from_state_dict(state_dict)

        return model

    model = model_architectures[architecture](*cfgs[architecture][quality], **kwargs)
    return model


def vmic(quality, metric="mse", pretrained=False, progress=True, **kwargs):
    r"""Swinv2 Transformer and Adaptive Global-inter and Channel-wise Context for Learned Image Compression

    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 6:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 6)')

    return _load_model("ab_test", metric, quality, pretrained, progress, **kwargs)

