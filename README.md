# Embodied XAI Is All You Need

**Revealing hidden preferences of vision–language models that leak into quadruped movement using XAI methods.**

*Explainable AI · Embodied AI — presented by S. Halim (University of Washington)*

### Embodied AI Agent instructed: *go to the friendly one*

<img width="1280" height="360" alt="friendly_combo_plain" src="https://github.com/user-attachments/assets/4361ddb8-d4ef-4cfc-87a8-3b2562b5bdc9" />

### 4D Saliency: *go to the friendly one*

<img width="1280" height="360" alt="friendly_combo_saliency" src="https://github.com/user-attachments/assets/ced274c9-7277-499b-8272-652152b06254" />

### Embodied AI Agent instructed: *go to the hostile one*

<img width="1280" height="360" alt="hostile_combo_plain" src="https://github.com/user-attachments/assets/6f4c4dd6-a50b-442a-9343-8624376a4887" />

### 4D Saliency: *go to the hostile one*

<img width="1280" height="360" alt="hostile_combo_saliency" src="https://github.com/user-attachments/assets/bd0dc629-69f7-429d-a829-acbd1806ffff" />

<img width="350" height="496" alt="poster_embodied_xai" src="https://github.com/user-attachments/assets/644988cf-7629-4456-a8fb-15701e5b0d73" />

> The full-resolution poster is here: [poster_embodied_xai.pdf](poster_embodied_xai.pdf).

---

## Motivation

Vision–language models (VLMs) are increasingly used as the "brain" that tells a
robot what to do from a camera view and a prompt. But VLMs quietly absorb human
associations from their training data — *red = danger*, *blue = calm* — and this
project asks a pointed question: **do those hidden associations leak into
physical motion when a VLM drives a legged robot, and can explainability (XAI)
tools detect the leak?**

The answer, from 50 trials per condition with a rigorous mirror-test design, is
**yes — mood-loaded language ("friendly" / "hostile") shifts which colored target
the agent walks to**, and **4D saliency** (integrated gradients projected through
depth into the 3D arena over time) reveals *where* the VLM's attention lands as
the decision unfolds.

<img width="800" height="450" alt="loco_iso" src="https://github.com/user-attachments/assets/9f33dbf0-de68-4ee8-b586-5e3e761d85e4" />

## Architecture

A slow semantic planner stacked on a fast locomotion controller, connected by a
continuous velocity command and an adapted dual-memory system.

<img width="828" height="261" alt="architecture" src="https://github.com/user-attachments/assets/dde9f1ee-3e90-478b-a612-848e2cb41dd8" />

| Layer | Component | Details |
|---|---|---|
| **Semantic loop (3 Hz)** | **Qwen2.5-VL-3B-Instruct** (frozen) | Reads the onboard camera + short-term video memory + long-term language memory. Emits a continuous `(v_x, v_y, w_z)` command, a one-sentence reasoning, a target choice, and a rewritten compressed memory — all in one JSON object. |
| | **Short-term video memory** | Last 4 onboard frames ~0.33 s apart, fed as a native Qwen video input (temporal position encoding). Adapted from MEM's dense visual window. |
| | **Long-term language memory** | A running natural-language summary (≤ 600 chars) that the VLM **rewrites and compresses** each step — drop failures, merge redundant events, replace-not-append. Adapted from MEM's trained recurrent summarizer. |
| | **Low-pass filter** (α = 0.35) | Smooths the planner's command between queries to prevent jerky transitions. |
| **Physical loop (50 Hz)** | **PPO gait policy** | A single actor–critic policy (`rsl-rl`, 512-256-128) trained across the full `(v_x, v_y, w_z)` command range so it tracks whatever the planner sends. |
| | **Unitree Go2** | Simulated quadruped on the Genesis physics engine. 12 actions (joint targets), 45-dim observation. |
| **XAI tap** | **Integrated gradients on chosen w_z** | Attributes the steering token the VLM *actually emitted* back to the input pixels. Combined with depth and camera pose, this projects into a **4D heat field** over the arena surfaces (floor, target faces, back wall). |

## MEM memory adaptation

The dual-memory design is adapted from **MEM** (Multi-Scale Embodied Memory,
Physical Intelligence, 2026) — a high/low-level policy architecture with a
trained video encoder and a trained recurrent language-memory summarizer.

Faithful MEM reproduction isn't possible here: MEM's video encoder requires
architecture surgery and large-corpus training; the language summarizer is
trained from an LLM-generated dataset; and the action expert is learned
end-to-end with flow matching. This project instead drives a **frozen,
off-the-shelf VLM** as the planner.

The adaptation keeps MEM's two-scale structure with a frozen-VLM fallback:

| MEM component | Faithful MEM | This project |
|---|---|---|
| High/low-level split | Trained π_HL, π_LL | Already present: VLM = slow planner (3 Hz), PPO = fast controller (50 Hz) |
| Short-term visual memory | Trained video encoder, K frames | Last 4 onboard frames fed as a **native Qwen video input** (temporal position encoding) |
| Long-term language memory | Trained recurrent summarizer w/ compression | The **same frozen VLM rewrites a compressed summary in-context each step**, prompted with MEM's compression rules |
| Memory update trigger | Every high-level step | Every VLM re-query (3 Hz) |

Both memory streams are **independently toggleable** (`MEM_SHORT_ENABLED`,
`MEM_LANG_ENABLED`) for MEM-style ablation — no-memory / only-video /
only-text / both.

## Experimental design

The test is built to be hard to fool:

- **N = 50 trials per condition** — worst-case 95 % CI half-width ±0.14.
- **Equal goals.** Two geometrically identical boxes, same forward distance, one
  left and one right.
- **Luminance matched.** Red and blue are sRGB luminance-matched so any skew
  can't be blamed on brightness.
- **Mirror test.** Red/blue sides are swapped across trials, separating color
  tracking from side tracking.
- **Identical-object control.** Both boxes red — measures the baseline side bias.
- **Mood prompts.** *"Go to the friendly one"* vs *"Go to the hostile one."*
- **Drift handling.** Trials that time out without entering the commit radius are
  labelled `drift` but committed to the nearest target, so every trial
  contributes to the statistics.

**V2 planner settings:** 4 recent frames ~0.33 s apart (no anchor frame),
long-term text memory ≤ 600 chars, queries at 3 Hz, temperature 0.7, command
low-pass α = 0.35, 6 s timeout, 0.7 m commit radius.

## Results

Results from **N = 50 trials per condition** (batched ×6 on A100), with Wilson
95 % confidence intervals and binomial tests against chance (0.5):

| Probe | Finding |
|---|---|
| **Color choice (mirror-pooled)** | "Friendly" → chose blue, "hostile" → chose red. The prompt word shifts color choice in the expected direction. |
| **Side bias (control)** | With identical targets the agent does *not* split 50/50 — there is a measurable positional bias, but the mirror design cancels it from the semantic conditions. |
| **Reach rate** | Substantially improved over the preliminary study; the MEM memory system and decisive-commit prompt reduce drift. |
| **Prompt-word effect** | Chi-square on the 2×2 (prompt × color) contingency table confirms that the word's valence shifts physical behavior. |

The full per-category breakdown, mirror decomposition, and chi-square test are
computed in §12d of the notebook.

## 4D saliency

The XAI analysis uses **integrated gradients (IG)** on the **chosen w_z steering
token** — the token the VLM *actually emitted* during the drive, not a
re-decoded greedy output. IG is computed with 8 Riemann steps (black baseline →
real image) and attributed back to the input pixel patches.

Each per-pixel saliency map is then **unprojected through the frame's depth
buffer** using the onboard camera's intrinsics and extrinsics (§5 camera
geometry): pixels that hit a surface (floor, target faces) become weighted 3D
points; sky/horizon pixels (no finite depth) are ray-intersected with the back
wall so no attention data is silently discarded. Accumulated over the trajectory,
this yields a **4D heat field** painted on the arena's real surfaces.

### MEM-architecture ablation (2 × 2 grid)

A saliency ablation reveals how each MEM component shapes which environmental
details drive the steering decision:

| IG input | No text memory | + Long-term text memory |
|---|---|---|
| **Single onboard frame** | `single_nomem` | `single_mem` |
| **Recent-frame video** | `video_nomem` | `video_mem` ⭐ (most faithful) |

All modes teacher-force the w_z the agent **actually chose** while driving.
The 20-trial drive simulation per category is run once and cached; each
ablation mode only adds its own attribution pass — no re-driving.

## Limitations

- **Single VLM.** All results come from one frozen Qwen2.5-VL-3B — findings may
  not transfer to other VLMs or model sizes.
- **Simulated environment.** A single Genesis scene with two colored boxes.
  Real-world transfer is untested.
- **MEM adaptation is approximate.** The frozen-VLM fallback lacks MEM's trained
  video encoder and robust summarizer; the language memory is produced in-context
  and may drift under distribution shift.
- **Attention ≠ cause.** IG shows *where* the model's gradients concentrate, which
  is necessary but not sufficient proof of *why* it chose.
- **No causal intervention.** The color–mood link is correlational; a
  counterfactual probe (e.g. swapping color channels inside the model) would
  strengthen the causal claim.

## Repository layout

```
.
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── .gitignore
├── architecture.svg                          # planner–controller architecture diagram
├── embodied_xai_paper_initial.pdf            # research paper
├── poster_embodied_xai.pdf                   # research poster
├── code_trace_4D_embodied_xai.ipynb          # full pipeline notebook (§0–§15)
└── simulation_demos/                         # cinematic renders & locomotion demos
    ├── cine_friendly_plain_combo.mp4
    ├── cine_friendly_sal_combo.mp4
    ├── cine_hostile_plain_combo.mp4
    ├── cine_hostile_sal_combo.mp4
    ├── cine_*_{iso,pov}.mp4                  # individual isometric / POV views
    ├── loco_combined.mp4
    ├── loco_iso.mp4
    └── loco_pov.mp4
```

The entire pipeline — environment, training, VLM planner, evaluation, saliency,
and cinematic rendering — lives in a single notebook
([`code_trace_4D_embodied_xai.ipynb`](code_trace_4D_embodied_xai.ipynb)),
organized into numbered sections (§0–§15). Each `%%writefile` cell exports a
standalone module (`go2_env.py`, `qre_utils.py`, `mem_planner.py`,
`go2_train.py`) so the code is reviewable without running Colab.

## Getting started

This was developed on **Google Colab with an A100 GPU** and headless EGL/Xvfb
rendering, using Google Drive for cross-session state.

1. Upload the notebook to Colab, or open it from
   [Drive](https://drive.google.com/drive/folders/1EBQKMpKgqXE6ueNPMsHTeDi4h6Ikmu-o?usp=sharing).
2. Set **Runtime → Change runtime type → A100 GPU**.
3. Run sections in order: **§0–§2** (runtime, Drive, dependencies) → **§3**
   (config) → **§4–§5** (env + utilities) → **§9** (PPO training or checkpoint
   restore) → **§10** (load policy + MEM planner) → **§11–§12** (closed-loop
   runtime + evaluation) → **§13–§15** (saliency + cinematic renders).

On Colab, **do not reinstall `torch`** — use the runtime's bundled CUDA build.
Outside Colab, install a matching `torch` first (see https://pytorch.org), then:

```bash
pip install -r requirements.txt
```

The locomotion stack pins `rsl-rl-lib==2.2.4` because Genesis targets that exact
API; other versions will break training.

## References & tools

1. **Genesis** — universal physics & rendering engine for robotics.
2. **Qwen2.5-VL-3B-Instruct** — vision–language model, Alibaba.
3. Schulman et al. — *Proximal Policy Optimization* (PPO), 2017.
4. **rsl-rl** — fast on-policy RL runner; **Unitree Go2** URDF.
5. **MEM** — *Multi-Scale Embodied Memory for Vision Language Action Models*,
   Physical Intelligence, 2026.

## Citation

If you use this work, please cite it via [`CITATION.cff`](CITATION.cff) or:

> S. Halim. *Embodied XAI Is All You Need: Revealing hidden preferences
> of vision–language models that leak into quadruped movement using XAI methods.*

## License

Released under the [MIT License](LICENSE).
