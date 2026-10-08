<div align="center">

# EMG-Parakeet

### Parameter-Efficient Adaptation of a Pretrained Neural Transducer for Surface Electromyography-to-Text Recognition

**Silent and vocalized sEMG → acoustic features → text**

[Getting started](#getting-started) · [Datasets](#datasets) · [Pretrained checkpoints](#pretrained-checkpoints) · [Inference](#inference-and-evaluation) · [Training](#training) · [Licensing](#licensing-and-attribution)

</div>

## Overview

EMG-Parakeet connects an sEMG-to-mel transduction model to the pretrained [NVIDIA Parakeet-TDT-1.1B](https://huggingface.co/nvidia/parakeet-tdt-1.1b) recognizer through a mel adapter and encoder adapters. Recognition uses predicted acoustic features directly, without waveform synthesis or external LLM correction.

The implementation supports **Digital Voicing (DV)** and **EMG-VCTK**, with separate evaluation scripts for silent and vocalized EMG. The default encoder adapter configuration is **8/64**: bottleneck dimensions of 8 and 64 for the lower and upper encoder layers.

## Getting started

```bash
git clone https://github.com/shcDE/EMG-Parakeet-Official.git
cd EMG-Parakeet-Official
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the Python dependencies in one command:

```bash
python -m pip install -r requirements.txt
```

This recipe targets **Python 3.11 on Linux x86_64 with an NVIDIA GPU**, using the maintainer's **PyTorch 2.5.0 / torchaudio 2.5.0 / torchvision 0.20.0 CUDA 12.1 builds**. The official PyTorch wheel index is declared in the file. A compatible NVIDIA driver is required; installing Python packages does not install a GPU driver. CPU-only, macOS and other CUDA builds require a separately matched PyTorch family rather than this CUDA-specific file. See [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/).

On Ubuntu, install system tools if they are not already available:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev libsndfile1 ffmpeg sox
```

For a non-default Python 3.11 installation, source builds may require its matching development headers. The system setup is separate from `pip`.

## Datasets

Download data from the original providers. Dataset files and their rights are separate from this repository's source-code license.

| Dataset | Official download | Reference | Terms |
| --- | --- | --- | --- |
| Digital Voicing / Silent Speech EMG | [Zenodo: 4064409](https://zenodo.org/records/4064409) | [Gaddy and Klein, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.445/) | CC BY 4.0 in the record metadata; retain attribution and applicable notices. |
| EMG-VCTK | [OSF: jbsu2](https://osf.io/jbsu2/) | [Scheck et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/scheck25_interspeech.html) | Custom terms: non-commercial research use, no re-identification, no public redistribution of the complete dataset; limited attributed excerpts are allowed. See [full terms](EMG-VCTK-TERMS.md). |

### Digital Voicing

Download `emg_data.tar.gz` from Zenodo and extract it locally. The loader expects the `silent_parallel_data`, `voiced_parallel_data` and `nonparallel_data` directories beneath the extracted `emg_data` directory. Use the release's original directory and session structure.

Set the path below to your extracted data. The other paths refer to files in this repository:

```bash
DV_ROOT="/absolute/path/to/emg_data"
DV_NORMALIZERS="$PWD/normalizers.pkl"

DV_ARGS=(
  "--silent_data_directories=$DV_ROOT/silent_parallel_data"
  "--voiced_data_directories=$DV_ROOT/voiced_parallel_data,$DV_ROOT/nonparallel_data"
  "--testset_file=$PWD/testset_largedev.json"
  "--text_align_directory=$PWD/text_alignments"
  "--normalizers_file=$DV_NORMALIZERS"
)
```

The default protocol uses `testset_largedev.json`. Changing to `testset_origdev.json` changes the validation protocol. Keep the split used for the released checkpoints.

Phoneme alignments are needed to reproduce stage-1 phoneme supervision and phoneme metrics. See the [upstream preparation instructions](https://github.com/dgaddy/silent_speech) and [alignment repository](https://github.com/dgaddy/silent_speech_alignments). Arrange the extracted files as `text_alignments/<session>/<session>_<index>_audio.TextGrid`. The current source also contains `text_alignments.tar.gz`. If using that archive, inspect its directory structure before extracting it into the repository root. Alignment files retain their original terms.

Missing TextGrid files are replaced by silence labels in the current loader. Consequently, a run can finish without reproducing the intended phoneme supervision. For text-only evaluation, `--skip_mel_metrics` disables the DTW/phoneme metric pass, but does not change the dataset loader's file requirements.

### EMG-VCTK

The [dataset terms](EMG-VCTK-TERMS.md), supplied by the maintainer, specify non-commercial research use and prohibit speaker re-identification. Do not publicly rehost the complete dataset on GitHub, Google Drive or another service. Internal sharing within a research group/institution is permitted under the same terms. Short audio clips or visualizations may be included in non-commercial research outputs with proper attribution to Scheck et al. (2025). This excerpt permission is not a blanket permission to redistribute arbitrary portions of the dataset.

Download the EMG recording sessions and associated metadata/audio from the official OSF project. Keep the original session directories and their `train`, `valid` and `test` subdirectories under a common `EMG-VCTK` root. The indexer looks for matching JSON metadata, `_emg.npy` files, `.wav` recordings and optional TextGrid files. **`TTS-Targets.zip` is not the EMG recording dataset and is not used by the recipe below.**

```bash
VCTK_ROOT="/absolute/path/to/EMG-VCTK"

python emg_vctk_scan.py \
  --emg_vctk_root "$VCTK_ROOT" \
  --scan_output emg_vctk_layout_local.json

python emg_vctk_index.py \
  --emg_vctk_root "$VCTK_ROOT" \
  --manifest_file emg_vctk_manifest_local.json
```

Generate the manifest on your own machine: the checked-in manifest contains the author's absolute paths. The indexer enforces prompt-disjoint splits by default, moving overlapping prompts into the stricter split (`test`, then `dev`, then `train`). Keep this policy and the same dataset release for reproduction; inspect the reported counts.

For the transfer recipe below, the DV normalizer is reused. Always use the exact normalizer paired with the published checkpoint; the presence of `normalizers_vctk.pkl` does not mean it can be substituted.

```bash
VCTK_ARGS=(
  "--manifest_file=$PWD/emg_vctk_manifest_local.json"
  "--normalizers_file=$PWD/normalizers.pkl"
)
```

The loader's defaults are 800 Hz raw EMG and eight channels. It excludes silent examples with no matching audible reference. Missing TextGrids also fall back to silence labels, so prepare/check alignments before reproducing phoneme-supervised training. Raw-EMG scaling options must match the checkpoint's training configuration.

## Select the dataset loader

**Training and silent evaluation currently select their dataset through Python imports.** Directory names such as `output_stage1_emg_vctk_ft` do not select a dataset, and there is no shared `--dataset` switch for these scripts.

| Script | Digital Voicing | EMG-VCTK |
| --- | --- | --- |
| `transduction_model.py` | `from read_emg import EMGDataset, SizeAwareSampler` | `from read_emg_vctk import EMGVCTKDataset as EMGDataset, SizeAwareSampler` |
| `train_parakeet.py` | `from read_emg import EMGDataset, SizeAwareSampler` | `from read_emg_vctk import EMGVCTKDataset as EMGDataset, SizeAwareSampler` |
| `evaluate_asr.py` | `from read_emg import EMGDataset` | `from read_emg_vctk import EMGVCTKDataset as EMGDataset` |
| `pretrain_mel_adapter.py` | `from read_emg import EMGDataset` | Not run in the EMG-VCTK transfer recipe. |
| `evaluate_asr_voiced.py` | Keep both imports. Pass `--dataset gaddy`. | Keep both imports. Pass `--dataset vctk --novctk_eval_silent_only`. |

For the first three scripts, enable **only one** of the two dataset import lines. Their EMG-VCTK alternatives are already present as comments. Switch all three consistently when moving between DV and EMG-VCTK: the evaluation and ASR-training scripts also import `transduction_model.py`.

Complete DV training/evaluation before switching to EMG-VCTK. If running the two corpora concurrently, use separate working copies to avoid changing imports beneath another run.

## Pretrained checkpoints

**Google Drive download:** The table below defines the intended release contents and extraction locations.

| Package | Google Drive | Required checkpoint paths after extraction | Evaluation modes |
| --- | --- | --- | --- |
| Digital Voicing and EMG-VCTK transfer, 8/64 each | [Drive for Checkpoints](https://drive.google.com/drive/folders/1yI5_JcUXf_LSMvfQceIpRwVSvYVOGZGK?usp=sharing) | `output_stage1_dv/model.pt`, `output_parakeet_dv_8_64/parakeet_asr_best.pt`, `output_stage1_emg_vctk_ft/model.pt`, `output_parakeet_emg_vctk_8_64_ft/parakeet_asr_best.pt`| Silent and voiced |

Each corpus uses the **same stage-1/ASR checkpoint pair** for silent and voiced evaluation. The evaluation script selects the recording mode. Training is unnecessary when the matching released checkpoints and supporting files are available.

The EMG-VCTK terms prohibit publicly redistributing the complete dataset. The checkpoint packages must not bundle those recordings. Trained weights are not expressly addressed in the supplied terms; provider clarification is still needed before declaring their distribution permitted.

Each download should include the stage-1 checkpoint, ASR adaptation checkpoint, exact feature normalizer, configuration/environment metadata, checksums and applicable attribution/license notices. Keep the provided split identifiers and reproduce the manifest locally. Original EMG/audio recordings should be downloaded separately from the providers.

`parakeet_asr_best.pt` is **not a standalone 1.1B recognizer**. In the current implementation, `ParakeetMelASR.save()` saves the mel adapter, trainable NeMo parameters and metadata. The frozen backbone is loaded separately from `nvidia/parakeet-tdt-1.1b`. Keep that upstream model identifier and the architecture flags consistent with the checkpoint.

## Inference and evaluation

First download/extract the matching checkpoints, prepare the dataset, define `DV_ARGS` or `VCTK_ARGS`, and select the appropriate imports above. These are **corpus evaluation commands**, not a single-file or live-streaming inference interface. Although recognition uses EMG-derived features, the current dataset loaders still expect the corpus metadata and associated recordings.

### Digital Voicing: silent

Use the DV imports.

```bash
python evaluate_asr.py "${DV_ARGS[@]}" \
  --models output_stage1_dv/model.pt \
  --parakeet_checkpoint output_parakeet_dv_8_64/parakeet_asr_best.pt \
  --adapter_lower_dim 8 --adapter_upper_dim 64 \
  --output_directory eval_test_parakeet_dv_8_64
```

### Digital Voicing: voiced

```bash
python evaluate_asr_voiced.py "${DV_ARGS[@]}" \
  --dataset gaddy \
  --models output_stage1_dv/model.pt \
  --parakeet_checkpoint output_parakeet_dv_8_64/parakeet_asr_best.pt \
  --adapter_lower_dim 8 --adapter_upper_dim 64 \
  --output_directory eval_test_parakeet_dv_8_64_voiced
```

### EMG-VCTK: silent

Switch the three shared scripts to the EMG-VCTK imports first.

```bash
python evaluate_asr.py "${VCTK_ARGS[@]}" \
  --vctk_eval_silent_only \
  --models output_stage1_emg_vctk_ft/model.pt \
  --parakeet_checkpoint output_parakeet_emg_vctk_8_64_ft/parakeet_asr_best.pt \
  --adapter_lower_dim 8 --adapter_upper_dim 64 \
  --output_directory eval_test_parakeet_emg_vctk_8_64_ft
```

### EMG-VCTK: voiced

**Both `--dataset vctk` and `--novctk_eval_silent_only` are required.** The dataset flag defaults to `gaddy`; disabling silent-only filtering alone does not select EMG-VCTK. Keep `--vctk_eval_silent_only` out of the shared `VCTK_ARGS` array to avoid conflicting mode flags.

```bash
python evaluate_asr_voiced.py "${VCTK_ARGS[@]}" \
  --dataset vctk \
  --novctk_eval_silent_only \
  --models output_stage1_emg_vctk_ft/model.pt \
  --parakeet_checkpoint output_parakeet_emg_vctk_8_64_ft/parakeet_asr_best.pt \
  --adapter_lower_dim 8 --adapter_upper_dim 64 \
  --output_directory eval_test_parakeet_emg_vctk_8_64_ft_voiced
```

### Outputs and options

| Item | Meaning |
| --- | --- |
| `eval_log.txt` | Evaluation configuration, progress and metrics. |
| `predictions.txt` | Normalized reference/hypothesis pairs (`REF:` / `HYP:`). |
| WER / CER | Corpus-level word/character error rates, printed as percentages. |
| `--dev` | Evaluate the development split instead of the default test split. |
| `--skip_mel_metrics` | Skip DTW loss and phoneme accuracy; retain ASR evaluation. |
| `--save_eval_mels` | Save mel tensors during evaluation. |
| `--oracle` | Feed ground-truth mel features to the ASR backend; this is a diagnostic reference condition, not EMG recognition. |

## Training

### 1. Digital Voicing

Enable DV imports and define `DV_ARGS` first. Keep the original split and matching feature normalization throughout all stages.

**Stage 1 — EMG-to-mel transduction**

```bash
python transduction_model.py "${DV_ARGS[@]}" \
  --output_directory output_stage1_dv
```

**Stage 2a — mel-adapter warm-up**

```bash
python pretrain_mel_adapter.py "${DV_ARGS[@]}" \
  --start_training_from output_stage1_dv/model.pt \
  --adapter_lower_dim 8 --adapter_upper_dim 64 \
  --output_directory output_meladapter_pretrain_dv_8_64
```

**Stage 2b — ASR adaptation**

```bash
python train_parakeet.py "${DV_ARGS[@]}" \
  --start_training_from output_stage1_dv/model.pt \
  --parakeet_checkpoint output_meladapter_pretrain_dv_8_64/parakeet_asr_pretrained.pt \
  --adapter_lower_dim 8 --adapter_upper_dim 64 \
  --nojoint_finetune \
  --nocreate_wandb_logger \
  --output_directory output_parakeet_dv_8_64
```

The code defaults to 80 stage-1 epochs, 10 warm-up epochs and 20 ASR-adaptation epochs. `--nojoint_finetune` keeps the transduction model fixed during ASR adaptation. The pretrained ASR backbone remains frozen while the mel adapter and encoder adapters are optimized. W&B logging is disabled in these examples; omit `--nocreate_wandb_logger` to use it.

After training, run the DV silent and voiced evaluation commands above before changing imports.

### 2. EMG-VCTK transfer from Digital Voicing

Switch the shared scripts to the EMG-VCTK imports and define `VCTK_ARGS` first.

**Stage 1 — fine-tune the transduction model from DV**

```bash
python transduction_model.py "${VCTK_ARGS[@]}" \
  --start_training_from output_stage1_dv/model.pt \
  --output_directory output_stage1_emg_vctk_ft
```

**Stage 2 — adapt from the DV ASR checkpoint**

```bash
python train_parakeet.py "${VCTK_ARGS[@]}" \
  --start_training_from output_stage1_emg_vctk_ft/model.pt \
  --parakeet_checkpoint output_parakeet_dv_8_64/parakeet_asr_best.pt \
  --adapter_lower_dim 8 --adapter_upper_dim 64 \
  --nojoint_finetune \
  --nocreate_wandb_logger \
  --output_directory output_parakeet_emg_vctk_8_64_ft
```

**Do not run an additional `pretrain_mel_adapter.py` warm-up on EMG-VCTK for this recipe.** It starts from the DV ASR adaptation checkpoint directly. “Full FT” here refers to fine-tuning the stage-1 transduction model; it does not mean unfreezing all 1.1B Parakeet parameters.

The training sampler uses different length units for the two loaders. `train_parakeet.py` documents `--emg_batch_tokens 8000` as an EMG-VCTK-specific budget, versus the default 256000 appropriate to DV raw-sample lengths. The commands above preserve the supplied recipe; use the released run configuration for exact batch settings. If changing this budget for memory reasons, record the change.

### Reproducibility notes

- Checkpoint, normalizer, model dimensions, dataset split and preprocessing options form one configuration. Do not mix files from different runs.
- The default EMG-VCTK training loader includes audible and silent recordings; development/test loading is silent-only unless explicitly changed for voiced evaluation.
- `parakeet_asr_best.pt` is saved on improved development WER. Stage-1 `model.pt` is overwritten each epoch in the current training loop; it is not automatically a best-development checkpoint.
- Use a new output directory for a new run. Keep logs, exact flags, package versions and the source commit with released checkpoints.
- Complete dataset evaluation and GPU training were not exercised as part of preparing this documentation. A clean-environment reproduction should be checked before declaring a release fully reproducible.

## Repository guide

| File | Purpose |
| --- | --- |
| `architecture.py`, `ropetransformer.py` | Stage-1 transduction architecture. |
| `read_emg.py` | Digital Voicing loader. |
| `read_emg_vctk.py` | EMG-VCTK loader. |
| `emg_vctk_scan.py`, `emg_vctk_index.py` | Inspect the EMG-VCTK layout and build a local manifest. |
| `transduction_model.py` | Train/fine-tune stage 1. |
| `pretrain_mel_adapter.py` | DV mel-adapter warm-up. |
| `train_parakeet.py` | ASR adaptation. |
| `parakeet_mel_asr.py` | Mel interface, Parakeet integration and adapter checkpoint I/O. |
| `evaluate_asr.py`, `evaluate_asr_voiced.py` | Silent and voiced corpus evaluation. |
| `asr_evaluation.py` | Transcript normalization and WER/CER computation. |

## Licensing and attribution

Original project code is covered by the repository's [MIT license](LICENSE), subject to retained third-party notices. **The source-code license does not relicense datasets, pretrained models, alignment resources or other third-party material.** See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

- **Upstream code:** acknowledge [David Gaddy's silent-speech implementation](https://github.com/dgaddy/silent_speech) and retain its MIT copyright/license notice for copied or adapted code. The mel-processing helpers also follow [HiFi-GAN](https://github.com/jik876/hifi-gan); retain the relevant MIT notice. Acknowledging a paper does not replace retaining license notices.
- **Digital Voicing data:** obtain it from the original Zenodo release and comply with CC BY 4.0 and any applicable resource-specific notices.
- **EMG-VCTK data:** follow the [full terms](EMG-VCTK-TERMS.md): non-commercial research use, no re-identification, no public redistribution of the complete dataset, internal sharing under the same terms, and limited attributed excerpts in non-commercial research outputs. The supplied terms do not expressly address trained checkpoint distribution; neither permission nor a blanket prohibition on weights is asserted here. Clarify that point with the provider before publishing the EMG-VCTK checkpoint package.
- **Parakeet-TDT-1.1B:** the upstream model card specifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Preserve applicable attribution, license links and change notices in redistributed model material; do not label NVIDIA weights as original MIT-licensed project code.
- **Model downloads:** checkpoint packages have their own release notices. Verify the dataset/model terms and the actual package contents before publishing a Google Drive link. Hosting files outside GitHub does not change their licensing obligations.

This implementation uses NVIDIA NeMo and credits the dataset and model authors. It is not endorsed by the upstream authors or organizations. Research evaluation does not establish clinical suitability.

## Citation

The paper title is **“EMG-Parakeet: Parameter-Efficient Adaptation of a Pretrained Neural Transducer for Surface Electromyography-to-Text Recognition.”** A complete paper citation will be added when bibliographic details are available.

Please also cite the datasets and upstream methods used in your experiment:

- Gaddy, D. and Klein, D. (2020). [Digital Voicing of Silent Speech](https://aclanthology.org/2020.emnlp-main.445/). EMNLP.
- Gaddy, D. and Klein, D. (2021). [An Improved Model for Voicing Silent Speech](https://aclanthology.org/2021.acl-short.23/). ACL-IJCNLP.
- Scheck, K., Dombeck, T., Ren, Z., Wu, P., Wand, M. and Schultz, T. (2025). [DiffMV-ETS: Diffusion-based Multi-Voice Electromyography-to-Speech Conversion using Speaker-Independent Speech Training Targets](https://www.isca-archive.org/interspeech_2025/scheck25_interspeech.html). Interspeech, 5573–5577. DOI: [10.21437/Interspeech.2025-1914](https://doi.org/10.21437/Interspeech.2025-1914).
- Follow the citations in the [Parakeet-TDT-1.1B model card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) for the pretrained recognizer and TDT/FastConformer methods.

For questions or reproducibility issues, please open a [GitHub issue](https://github.com/shcDE/EMG-Parakeet-Official/issues).
