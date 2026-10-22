# ProCal: Inference-Time Proposal Calibration for Open-Vocabulary Object Detection

**Jae-Ryung Hong, Ho-Joong Kim, Seong-Whan Lee**  
Department of Artificial Intelligence, Korea University

[Paper][paper] | [PDF][paper-pdf] | [Code][repo] | [Citation](#citation)

> **README Draft**  
> This draft is based on a comparison of the paper and the public repository. The installation steps and GPU evaluation have not been executed as part of preparing this README. The evaluation entry point still requires attention, including argument propagation. Please read the [Implementation Notes](#implementation-notes) before using the commands below.

## Overview

**ProCal** is a training-free method that calibrates the novel-category scores of an open-vocabulary detector at inference time. It combines foreground and background cues from a frozen vision-language model (VLM) into a proposal prior and uses this prior to adjust the detector's novel-category scores. It does not modify proposal generation or box regression. Please refer to the [paper][paper-pdf] for the full method and experimental details.

This repository uses a **CLIPSelf / F-ViT** detector. The main evaluation entry point is [`F-ViT/test.py`](F-ViT/test.py). The ProCal workflow in the public implementation is:

```text
Set up the environment and datasets
  -> Prepare detector and backbone weights
  -> Evaluate the baseline
  -> Save per-proposal VLM similarities to JSON files
  -> Load the cached similarities, apply ProCal, and evaluate
```

ProCal itself requires no additional training, but computing the proposal priors requires additional VLM inference.

<!-- Add the original Figure 2 image to assets/procal_overview.png, then uncomment the line below.
![ProCal overview](assets/procal_overview.png)
-->

## Main Results

The following numbers are **reported in Tables I and II of the paper**. They were not reproduced while preparing this README and should not be interpreted as verified outputs of the current public code under its default settings.

### OV-COCO

| Backbone | Method | Base AP50 | Novel AP50 | All AP50 |
|:--|:--|--:|--:|--:|
| ViT-B/16 | CLIPSelf + F-ViT | 54.8 | 37.4 | 50.3 |
| ViT-B/16 | CLIPSelf + F-ViT + ProCal | 54.3 | **40.1 (+2.7)** | 50.5 |
| ViT-L/14 | CLIPSelf + F-ViT | 64.2 | 44.5 | 59.0 |
| ViT-L/14 | CLIPSelf + F-ViT + ProCal | 63.4 | **45.7 (+1.2)** | 58.8 |

### OV-LVIS

| Backbone | Method | APr | AP |
|:--|:--|--:|--:|
| ViT-B/16 | CLIPSelf + F-ViT | 25.8 | 26.4 |
| ViT-B/16 | CLIPSelf + F-ViT + ProCal | **26.9 (+1.1)** | 26.0 |
| ViT-L/14 | CLIPSelf + F-ViT | 36.8 | 37.3 |
| ViT-L/14 | CLIPSelf + F-ViT + ProCal | **39.3 (+2.5)** | 37.3 |

AP50 denotes box AP at an IoU threshold of 0.5. For LVIS, AP and APr are averaged over IoU thresholds from 0.5 to 0.95, and APr measures performance on rare categories. Values in parentheses indicate changes relative to the baseline with the same backbone. Source: [paper][paper-pdf].

## Installation

### Environment and Dependencies

The main versions recorded in [`ProCal_requirement.txt`](ProCal_requirement.txt) are listed below. This file is an **environment snapshot**, not a minimal requirements file verified to install successfully in a clean environment.

| Component | Recorded Version / Identifier |
|:--|:--|
| PyTorch | `1.13.1` |
| torchvision | `0.14.1+cu117` |
| MMCV | `mmcv-full==1.7.2` |
| MMDetection | `2.28.2` |
| NumPy | `1.26.4` |
| LVIS API | `0.5.3` |
| OpenCLIP implementation | CLIPSelf commit `1c7fe9c5c38d800903d4754a89d7a8fcc7977101` |

The Python version, operating system, and GPU specifications cannot be determined from this file alone. The CUDA toolkit and driver versions should also be checked separately from the CUDA tag in the `torchvision` wheel name.

### Repository and OpenCLIP Setup

```bash
git clone https://github.com/Jar199/ProCal.git
cd ProCal
```

Create a separate virtual environment and first install PyTorch, torchvision, MMCV, and MMDetection compatible with the versions above. Refer to the [F-ViT requirements](F-ViT/README.md#requirements) for the underlying setup instructions, but note that the versions listed there differ from this repository's environment snapshot.

The detector uses the modified `open_clip` implementation from CLIPSelf. Install the revision specified in the environment snapshot as follows:

```bash
# Run after setting up PyTorch, torchvision, MMCV, and MMDetection.
python -m pip install -e \
  "git+https://github.com/wusize/CLIPSelf@1c7fe9c5c38d800903d4754a89d7a8fcc7977101#egg=open_clip_torch"

python -m pip check
```

**Note:** The root-level `setup.py` references `src/open_clip/version.py`, but that file was not found at the inspected public path. Prepare the CLIPSelf dependency separately as described above rather than immediately running `pip install -e .` at the root of this repository. Installing the entire unmodified `ProCal_requirement.txt` also requires separate validation.

<!-- Before release: add the actual Python, OS, CUDA, and GPU specifications, along with minimal installation commands verified in a clean environment. -->

## Dataset Preparation

Run the evaluation commands from **`ProCal/F-ViT/`**. The current configs load images and annotations from `data/` at the repository root. Follow the paths used by the actual configs below rather than the example layout in the inherited F-ViT README.

```text
ProCal/
|-- data/
|   |-- Annotations/
|   |   |-- instances_val2017_all_2.json
|   |   |-- lvis_v1_val.json
|   |   |-- instances_train2017_seen_2_65_cat.json  # For detector training
|   |   `-- lvis_v1_train_seen_1203_cat.json        # For detector training
|   `-- Images/
|       |-- train2017/
|       `-- val2017/
`-- F-ViT/
    |-- datasets/            # Category splits and class embedding paths
    `-- checkpoints/        # Model weights to download separately
```

Obtain COCO and LVIS images and annotations from their respective dataset distribution sources. For annotations with open-vocabulary splits, refer to [F-ViT Data Preparation](F-ViT/README.md#data-preparation). Image directories may be symbolic links to an existing dataset installation.

| Evaluation Dataset | `data.test.ann_file` | `data.test.img_prefix` |
|:--|:--|:--|
| OV-COCO | `../data/Annotations/instances_val2017_all_2.json` | `../data/Images/val2017/` |
| OV-LVIS | `../data/Annotations/lvis_v1_val.json` | `../data/Images/` |

For LVIS, ensure that the `train2017/` and `val2017/` images referenced by the annotations are available under the corresponding subdirectories. Detector-training annotations are not needed for evaluation only.

Category metadata and class embedding paths are specified by fields such as `seen_classes`, `all_classes`, and `class_embed` in each config. Detector embeddings are referenced relative to **`F-ViT/datasets/embeddings/`**; do not confuse this location with the root-level `data/embeddings/` directory. When changing dataset locations, update the paths actually used by the loader through the config or `--cfg-options`.

Reference configs: [OV-COCO](F-ViT/configs/ov_coco/fvit_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_original.py), [OV-LVIS](F-ViT/configs/ov_lvis/fvit_vitl14_upsample_fpn_bs64_4x_ovlvis_eva_original.py).

## Pretrained Models

### Detector and Backbone

Prepare the **detector checkpoint (`.pth`)** and the **CLIPSelf backbone checkpoint (`.pt`)** separately. Backbone construction also uses a pretrained-weight path, so providing the detector checkpoint alone does not necessarily satisfy all initialization requirements.

The filenames below are taken from the repository's [F-ViT model documentation](F-ViT/README.md#detectors) and configs. All paths are relative to `F-ViT/checkpoints/`.

| Dataset | Backbone | Detector Checkpoint | CLIPSelf Backbone Checkpoint |
|:--|:--|:--|:--|
| OV-COCO | ViT-B/16 | `fvit_eva_vitb16_ovcoco_clipself_proposals.pth` | `eva_vitb16_coco_clipself_proposals.pt` |
| OV-COCO | ViT-L/14 | `fvit_eva_vitl14_ovcoco_clipself_proposals.pth` | `eva_vitl14_coco_clipself_proposals.pt` |
| OV-LVIS | ViT-B/16 | `fvit_eva_vitb16_ovlvis_clipself_patches.pth` | `eva_vitb16_lvis_clipself_patches.pt` |
| OV-LVIS | ViT-L/14 | `fvit_eva_vitl14_ovlvis_clipself_patches.pth` | `eva_vitl14_lvis_clipself_patches.pt` |

Download links are provided under [Detector Checkpoints](F-ViT/README.md#test) and [CLIPSelf Checkpoints](F-ViT/README.md#clipself-checkpoints). These links refer to the underlying model releases. **The correspondence between these files, their configurations, and the exact experiments in the paper must be confirmed before the final release.**

### VLM for ProCal

[`F-ViT/apis_test.py`](F-ViT/apis_test.py) initializes the model used for prior computation with the following identifiers:

| Setting | Value |
|:--|:--|
| Model | `ViT-B-16` |
| Pretrained | `openai` |
| API | `open_clip.create_model_and_transforms` |

This VLM is separate from the detector backbone, including when the detector uses ViT-L/14. Ensure that the weights can be downloaded on the first run, or prepare a compatible cache for the OpenCLIP implementation in use.

## Evaluation

> **Before Running**  
> The commands below are draft examples aligned with the actual argument parser and execution branches. The argument-propagation issue in the public evaluation entry point must be resolved first. Do not treat these commands as ready to run without reviewing the [Implementation Notes](#implementation-notes).

### 1. Common Setup

The following example uses OV-COCO with ViT-B/16. Run the subsequent commands in the same terminal so that these shell variables remain available.

```bash
# From the repository root:
cd F-ViT

CONFIG="configs/ov_coco/fvit_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals.py"
CHECKPOINT="checkpoints/fvit_eva_vitb16_ovcoco_clipself_proposals.pth"
PRIOR_DIR="proposal_prior/ovcoco/vitb16"
RUN_DIR="work_dirs/ovcoco_vitb16"
GPUS=1

mkdir -p "$RUN_DIR" "$PRIOR_DIR"
```

The positional arguments to `dist_test.sh` are **config, checkpoint, and number of GPUs**, in that order. The VLM used to generate priors is initialized in `multi_gpu_test()`, so these instructions use the distributed launcher even for a single GPU by passing `1` as the GPU count.

### 2. Evaluate the Baseline

In the current parser, **omitting** `--baseline` leaves `baseline=True`.

```bash
bash dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" \
  --work-dir "$RUN_DIR/baseline" \
  --out "$RUN_DIR/baseline.pkl" \
  --eval bbox
```

### 3. Cache Similarities for Proposal Priors

In the current implementation, the prior-generation branch saves similarities and then returns `None` instead of detections for that image. Cache generation and evaluation must therefore be run as separate stages.

```bash
bash dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" \
  --baseline \
  --load_proposal_prior \
  --prior_folder "$PRIOR_DIR" \
  --work-dir "$RUN_DIR/prior_generation" \
  --out "$RUN_DIR/prior_generation.pkl"
```

In this command, `--baseline` and `--load_proposal_prior` each set their corresponding value to **False**. Omit `--save_proposal_prior` to keep saving enabled. Do not add `--eval bbox` during cache generation. The `prior_generation.pkl` output satisfies the parser's output requirement; it is not a prediction file for evaluating detection performance.

The actual cache is saved to:

```text
F-ViT/proposal_prior/ovcoco/vitb16/<image_id>/proposal_prior.json
```

Each JSON file contains three similarity arrays, `bnk_sim`, `bnk_sim4`, and `bg_bnk_sim`, rather than a single final prior. The final prior is computed in the bbox head during evaluation.

### 4. Evaluate ProCal Using the Cached Similarities

```bash
bash dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" \
  --baseline \
  --prior_folder "$PRIOR_DIR" \
  --work-dir "$RUN_DIR/procal" \
  --out "$RUN_DIR/procal.pkl" \
  --eval bbox
```

At this stage, omit `--load_proposal_prior` to retain `load_proposal_prior=True`. The commands use the parser's numeric default of `0.2` for `gamma`.

Use the **same checkpoint and evaluation config** for the baseline and ProCal. Keep separate caches for each dataset, backbone, checkpoint, and proposal configuration. Regenerate the cache after any change that affects proposal counts, coordinates, or ordering.

### Other Datasets and Backbones

Replace the common variables with one of the configurations below, then follow the same steps. Use `--eval bbox` for OV-LVIS as well when comparing against the paper's box AP results.

<details>
<summary>OV-COCO / ViT-L/14</summary>

```bash
CONFIG="configs/ov_coco/fvit_vitl14_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals.py"
CHECKPOINT="checkpoints/fvit_eva_vitl14_ovcoco_clipself_proposals.pth"
PRIOR_DIR="proposal_prior/ovcoco/vitl14"
RUN_DIR="work_dirs/ovcoco_vitl14"
mkdir -p "$RUN_DIR" "$PRIOR_DIR"
```

</details>

<details>
<summary>OV-LVIS / ViT-B/16</summary>

```bash
CONFIG="configs/ov_lvis/fvit_vitb16_upsample_fpn_bs64_4x_ovlvis_eva_clipself_patches.py"
CHECKPOINT="checkpoints/fvit_eva_vitb16_ovlvis_clipself_patches.pth"
PRIOR_DIR="proposal_prior/ovlvis/vitb16"
RUN_DIR="work_dirs/ovlvis_vitb16"
mkdir -p "$RUN_DIR" "$PRIOR_DIR"
```

</details>

<details>
<summary>OV-LVIS / ViT-L/14</summary>

```bash
CONFIG="configs/ov_lvis/fvit_vitl14_upsample_fpn_bs64_4x_ovlvis_eva_clipself_patches.py"
CHECKPOINT="checkpoints/fvit_eva_vitl14_ovlvis_clipself_patches.pth"
PRIOR_DIR="proposal_prior/ovlvis/vitl14"
RUN_DIR="work_dirs/ovlvis_vitl14"
mkdir -p "$RUN_DIR" "$PRIOR_DIR"
```

</details>

### Outputs

`--out` specifies the prediction pickle path. When `--eval` is used, evaluation metrics are saved as JSON under `--work-dir`. The current JSON filename suffix contains a hard-coded string, so record the command and config alongside the output rather than inferring the experiment settings from the filename alone.

## Configuration

### Current Command-Line Semantics

The three boolean options below all use `action='store_false'`. Their behavior is therefore the opposite of a conventional enabling flag.

| Option | When Omitted | When Specified | Purpose |
|:--|:--:|:--:|:--|
| `--baseline` | `True` | `False` | Disables the baseline branch when specified |
| `--load_proposal_prior` | `True` | `False` | Disables cache loading when specified |
| `--save_proposal_prior` | `True` | `False` | Disables prior saving when specified |
| `--prior_folder` | `proposal_prior/ovcoco/vitb16` | Supplied path | Directory for per-image JSON caches |
| `--gamma` | Numeric `0.2` | Currently parsed as a string | Score-calibration strength; fix type conversion before overriding |

Source: [`F-ViT/test.py`](F-ViT/test.py).

### Key Implementation Settings

| Setting | Current Value | Location |
|:--|:--|:--|
| Localization logit scale | `50` | `models/fvit_head.py` |
| Background logit scale | `10` | `models/fvit_head.py` |
| Cue mixing weight | `lambda_ = 0.5` | `models/fvit_head.py` |
| Novel-score calibration strength | Default `gamma = 0.2` | `test.py` -> `models/fvit_head.py` |
| Final NMS IoU threshold | COCO `0.3` / LVIS `0.5` | Respective `*_eva_original.py` configs |
| RPN proposal limit | `max_per_img = 1000` | `model.test_cfg.rpn` |

The RPN proposal limit is different from the number of final output detections. The inspected code uses the proposals returned by the RPN rather than exposing a separate ProCal-specific `top-k` argument. Also, matching the logit scales and mixing weights does not guarantee equivalence to the paper's prior if the **order of activation and aggregation** differs. See the implementation notes below.

## Code Structure

| File / Directory | Role |
|:--|:--|
| [`F-ViT/test.py`](F-ViT/test.py) | Argument parsing, model and data loading, and evaluation output |
| [`F-ViT/dist_test.sh`](F-ViT/dist_test.sh) | Distributed evaluation launcher |
| [`F-ViT/apis_test.py`](F-ViT/apis_test.py) | Evaluation loop, prior VLM initialization, and text feature generation |
| [`F-ViT/models/fvit.py`](F-ViT/models/fvit.py) | RPN proposals, crop feature extraction, and similarity cache writing/loading |
| [`F-ViT/models/fvit_head.py`](F-ViT/models/fvit_head.py) | Proposal prior construction and novel-category score calibration |
| [`F-ViT/models/evaclip_vit.py`](F-ViT/models/evaclip_vit.py) | CLIPSelf / EVA-CLIP detector backbone |
| [`F-ViT/configs/ov_coco/`](F-ViT/configs/ov_coco/) | OV-COCO configs |
| [`F-ViT/configs/ov_lvis/`](F-ViT/configs/ov_lvis/) | OV-LVIS configs |
| [`F-ViT/datasets/`](F-ViT/datasets/) | Dataset loaders, category metadata, and embedding paths |

## Implementation Notes

The notes below come from a static review of the source. They do not imply that the code has been modified or that GPU reproduction experiments have been performed. They distinguish documentation issues from issues that require changes to the implementation.

<details>
<summary><strong>Checks required before evaluation and paper reproduction</strong></summary>

**Argument propagation.** In `test.py`, `args` is local to `main()`. However, `apis_test.py`, `models/fvit.py`, and `models/fvit_head.py` reference `args` without receiving or defining it. This connection must be resolved for the inspected `dist_test.sh -> test.py` execution path.

**Numeric arguments.** `--gamma` is not declared with `type=float`. Unlike the numeric default `0.2`, a value supplied on the command line is parsed as a string. Convert it to a number in the parser or at the call site before using it in arithmetic. If the boolean flag semantics are revised, update the commands above accordingly.

**Proposal prior formula.** The current implementation under the arithmetic-mean comment in `fvit_head.py` computes `sigmoid(lambda * 50 * loc + (1 - lambda) * 10 * bg)`. This is generally not equivalent to the form in Eq. (6) of the paper, `lambda * sigmoid(50 * loc) + (1 - lambda) * sigmoid(10 * bg)`. Here, `bg` denotes the negated background similarity, as in the code. Confirm the computation used for the paper's experiments, then align the implementation and documentation.

**Prompt strings.** In `apis_test.py`, the two object-related prompts use `a object`, whereas the paper uses `an object`. Match the strings to the version used in the experiments rather than silently correcting them.

**Cache-generation path.** Prior generation runs through `multi_gpu_test()`, and `fvit.py` returns `None` after generation. Do not treat this output as baseline or ProCal evaluation results. Also verify that the scaling of proposal coordinates applied to the original image and the crop preprocessing match the experimental implementation.

**Experiment-to-file mapping.** Do not automatically assume that the inherited F-ViT weight links, the reported results, and the current configs refer to identical experiments. Provide exact checkpoint identifiers and evaluation logs with the final release.

</details>

## Citation

Please cite the following paper when using this work:

```bibtex
@article{hong2026procal,
  title   = {{ProCal}: Inference-Time Proposal Calibration for Open-Vocabulary Object Detection},
  author  = {Hong, Jae-Ryung and Kim, Ho-Joong and Lee, Seong-Whan},
  journal = {arXiv preprint arXiv:2607.01759},
  year    = {2026},
  doi     = {10.48550/arXiv.2607.01759},
  url     = {https://arxiv.org/abs/2607.01759}
}
```

## Acknowledgements

We thank the authors of [CLIPSelf][clipself], [OpenCLIP][openclip], [EVA-CLIP][evaclip], [MMDetection][mmdet], and [MMCV][mmcv] for releasing the code and pretrained models on which this implementation builds.

## License and Contact

The root-level [`LICENSE`](LICENSE) specifies the **MIT License**. Please also consult the original distributions for the licenses and notices applicable to the underlying code, model weights, and datasets. CLIPSelf's licensing information is available in the [upstream README][clipself-license].

For questions and reproduction issues, open a [GitHub Issue][issues]. Include the commit, environment, command, config, checkpoint information, and error logs.

<!--
Pre-release checklist for maintainers
- Resolve argument propagation and finalize the public defaults and boolean flag interface.
- Compare the prior formula, prompt strings, and crop coordinate conventions with the experimental code.
- Verify the exact Python, CUDA, and GPU environment and minimal requirements in a clean setup.
- Provide checkpoint filenames/hashes and evaluation logs corresponding to the paper's tables.
- Explain the discrepancy in ViT-B/16 OV-COCO results between Tables I/V and Table III.
- Add the original figure asset and uncomment the Overview image.
- Update the draft warning and implementation notes once these issues have been resolved.
-->

[paper]: https://arxiv.org/abs/2607.01759
[paper-pdf]: https://arxiv.org/pdf/2607.01759v1
[repo]: https://github.com/Jar199/ProCal
[issues]: https://github.com/Jar199/ProCal/issues
[clipself]: https://github.com/wusize/CLIPSelf
[clipself-license]: https://github.com/wusize/CLIPSelf/blob/1c7fe9c5c38d800903d4754a89d7a8fcc7977101/README.md#license
[openclip]: https://github.com/mlfoundations/open_clip
[evaclip]: https://github.com/baaivision/EVA/tree/master/EVA-CLIP
[mmdet]: https://github.com/open-mmlab/mmdetection
[mmcv]: https://github.com/open-mmlab/mmcv
