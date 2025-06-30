import string
import torch
import blosc2
import os
import json
import numpy as np
from collections.abc import Iterator


def generate_tags() -> Iterator[str]:
    """Generate chain tags.

    Yields
    ------
    str
        The next chain tag

    """
    for i in range(1, 4):
        for j in range(len(string.ascii_uppercase) ** i):
            tag = ""
            for k in range(i):
                tag += string.ascii_uppercase[
                    j
                    // (len(string.ascii_uppercase) ** k)
                    % len(string.ascii_uppercase)
                ]
            yield tag


def blosc2_save(tnr: torch.Tensor, name: str, output_dir: os.PathLike):
    arr = tnr.detach().cpu().numpy().astype(np.float16)
    compressed = blosc2.compress2(
        arr.tobytes(),
        typesize=2,
        clevel=9,
        codec=blosc2.Codec.ZSTD,
        filters=[blosc2.Filter.BITSHUFFLE]
    )

    # Save shape & dtype info
    meta = {
        "shape": arr.shape,
        "dtype": "float16"
    }

    out_dir = output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "metadata.blosc2.json").open("w") as f:
        json.dump(meta, f)

    with (out_dir / f"{name}.blosc2").open("wb") as f:
        f.write(compressed)

