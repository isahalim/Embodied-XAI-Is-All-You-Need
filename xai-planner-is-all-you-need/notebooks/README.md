# Notebooks

The project runs as four notebooks in sequence. They were authored on **Google
Colab with an A100 GPU**, using headless EGL/Xvfb rendering and Google Drive for
cross-session state (checkpoints, configs, trial logs, videos). Each notebook
reloads the shared modules and any prior outputs from Drive, so a fresh runtime
can resume where the last one stopped.

The `.py` files here are the Colab exports of the notebooks — readable and
diffable. To run them, open the matching `.ipynb` in Colab (or convert with
`jupytext --to notebook nbX_*.py`), set the runtime to an A100 GPU, and execute
top to bottom.

| Notebook | Role | Writes |
|---|---|---|
| `nb0_smoke_test_and_config` | Empirical-verification stage. Confirms the A100/CUDA, headless rendering, Genesis rendering of the Go2 and the two-goal bias arena, camera depth + intrinsics/extrinsics, and that Qwen2.5-VL loads, obeys the JSON contract, exposes a single-token `turn`, and yields a reshapeable attention map. | `go2_env.py`, `qre_utils.py`, `config.py`, grid texture → Drive |
| `nb1_train_locomotion` | Trains one robust PPO locomotion policy with a wide command sampler so it can follow whatever direction the planner later sends. Ends with a high-quality eval video. | `go2_train.py`, `go2_eval.py`, checkpoints → Drive |
| `nb2_vlm_integration` | Closes the loop: Qwen2.5-VL plans, the PPO policy walks. Builds the luminance-matched red-vs-blue arena and validates the loop on an unambiguous **named-target control** before any bias prompt. | demo videos, VLM reasoning logs → Drive |
| `nb3_behavioral_bias` | The statistical core. Runs the fast decision harness across conditions with the mirror test, the identical-object control, and both prompt modes (`reason`, `action`). Computes χ², Cramér's V, Wilson 95% CIs, and separates explicit from implicit bias. | `trials.jsonl`, `trials_detailed.jsonl`, `*_stats.json` → Drive |

A fifth stage — **NB4**, a 3D temporal saliency map linking attention to
behavior across the whole path — is referenced throughout as the mechanistic
follow-up and is listed under future work in the [README](../README.md).

## Relationship to `src/`

The shared modules in [`../src`](../src) are the extracted, version-controlled
copies of the files that NB0 and NB1 generate via `%%writefile`. Editing the
modules under `src/` and uploading them to your Drive `config/` folder is the
clean way to change pipeline behavior without editing notebook string literals.
Where a module was written more than once across cells, `src/` keeps the final
version the pipeline actually used.
