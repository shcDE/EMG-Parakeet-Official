# Third-party notices

This document preserves upstream notices for the silent-speech and mel-processing code adapted in this project. Project modifications do not remove upstream copyrights. Dataset, alignment, model and dependency licenses remain separate. This is not an exhaustive provenance audit.

## David Gaddy — silent_speech

Source: https://github.com/dgaddy/silent_speech

```text
MIT License

Copyright (c) 2021 David Gaddy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Jungil Kong — HiFi-GAN

Source: https://github.com/jik876/hifi-gan

```text
MIT License

Copyright (c) 2020 Jungil Kong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## NVIDIA Parakeet and NeMo

The pretrained recognizer is obtained separately from [NVIDIA Parakeet-TDT-1.1B](https://huggingface.co/nvidia/parakeet-tdt-1.1b), jointly developed by NVIDIA NeMo and Suno.ai. Its model card specifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). EMG-Parakeet adds a mel interface and trains adaptation parameters for EMG-derived features. Preserve applicable model notices and identify these changes in any released adaptation package. The repository MIT license does not replace the upstream model license.

NeMo and other installed dependencies retain their own licenses; see the packages' distributions and [NVIDIA NeMo](https://github.com/NVIDIA/NeMo).

## Data and alignment resources

- [Digital Voicing / Silent Speech EMG](https://zenodo.org/records/4064409), David Gaddy, 2020: CC BY 4.0 according to the [record metadata](https://zenodo.org/api/records/4064409).
- [EMG-VCTK](https://osf.io/jbsu2/), Cognitive Systems Lab, University of Bremen: see the [full terms supplied by the maintainer](EMG-VCTK-TERMS.md). They specify non-commercial research use, prohibit re-identification and public redistribution of the complete dataset, allow internal sharing under the same terms, and allow limited excerpts in non-commercial research outputs with attribution. Cite Scheck et al. (2025). The supplied terms do not expressly address trained model/checkpoint distribution; seek provider clarification rather than presenting it as automatically permitted or prohibited.
- [Silent-speech alignments](https://github.com/dgaddy/silent_speech_alignments): inspect the original distribution terms before redistributing alignment files. This notice does not assign those files a new license.

The repository currently includes dataset-related manifests, split files, normalizers and an alignment archive. Retain applicable attribution and review each resource's provenance and redistribution terms before release. A source-code MIT license alone does not establish permission for those assets. Do not treat an absent or unclear asset license as unrestricted permission.
