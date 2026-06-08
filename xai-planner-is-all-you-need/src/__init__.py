"""Version-controlled copies of the shared library modules used by the pipeline.

These files are the canonical source for what notebooks NB0 and NB1 historically
wrote to Google Drive via `%%writefile` cells:

    config.py      shared constants (paths, camera, command ranges, VLM, conditions)
    qre_utils.py   render profile, sanity gate, camera geometry, bias arena,
                   Qwen2.5-VL load / JSON-contract parser / turn-token attention
    go2_env.py     Genesis Go2 environment (gym-style, 12 actions, 45-dim obs, 50 Hz)
    go2_train.py   PPO config + training entry point (rsl-rl)
    go2_eval.py    high-quality eval video with a trailing follow-camera

See ../notebooks/README.md for how these are loaded at runtime on Colab.
"""
