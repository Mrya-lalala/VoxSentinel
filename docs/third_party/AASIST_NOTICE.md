# AASIST third-party notice and provenance

This repository contains an **embedding-adapted** AASIST research head. The
graph attention blocks in `src/detectors/aasist_blocks.py` are ported from the
official AASIST implementation.

## Upstream source

- Project: AASIST - https://github.com/clovaai/aasist
- Commit: `a04c9863f63d44471dde8a6abcb3b082b07cd1d1`
- Paper: https://arxiv.org/abs/2110.01200
- Files inspected (SHA-256 of the fetched copies):

| Upstream path | SHA-256 |
| --- | --- |
| `models/AASIST.py` | `9E0D3E80937DD0577BEEA7883098465A479DA23A198EBC0D712ABCC59B0BEC50` |
| `config/AASIST.conf` | `C25023331685027CCE90E1B9A0D2DF10AA04B2A27D9B27D5AFA36E6815B0FE76` |
| `LICENSE` | `DA2E79B8592D166EF505224300968B80EBE1E4C217C43B94A5EC627D81CD4142` |
| `NOTICE` | `70EDB07F04DDC88E7155D6E826495E5EF99C3B1F0BD4E1B7FA35897B3854DBD0` |

## What is reused

Ports of `GraphAttentionLayer`, `HtrgGraphAttentionLayer` and `GraphPool` from
`models/AASIST.py`, with the modifications listed in the module docstring:

- dropout probabilities are constructor arguments (upstream hardcoded 0.2 for
  attention input dropout and 0.3 for graph pooling) and dropout is non-inplace;
- `_apply_BN` uses `reshape` so non-contiguous inputs are accepted;
- master nodes are passed explicitly as `[B, 1, dim]`.

The attention math, learned temperature handling, `BatchNorm1d`, `SELU`,
type-masked heterogeneous attention and top-k graph pooling are unchanged.

## What is not reused

- `CONV` (sinc/mel filterbank), `Residual_block` and the 2-D waveform encoder:
  this pipeline consumes frozen `[B, T, 1024]` embeddings and has no waveform or
  time-frequency front-end.
- The upstream training stack, dataset code and the ASVspoof2019 t-DCF package.
  The t-DCF package is distributed under CC BY-NC-SA 4.0 and is **not** copied
  or imported here.

## Upstream licence

```
AASIST
Copyright (c) 2021-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

The upstream `NOTICE` file additionally lists MIT-licensed subcomponents
(ASVspoof 2021 `Baseline-RawNet2`, `Jungjee/RawNet`) and the CC BY-NC-SA 4.0
ASVspoof2019 t-DCF package. None of those subcomponents are copied here.

## Scope note

This adaptation is a research head evaluated on frozen IndicWav2Vec embeddings.
It is not a reproduction of the waveform AASIST and no accuracy improvement
over the GRU baseline is claimed or established.
