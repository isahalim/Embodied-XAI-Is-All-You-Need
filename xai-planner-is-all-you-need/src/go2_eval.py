import argparse, os, pickle, torch
import numpy as np
from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from go2_env import Go2Env

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", type=str, default="go2_locomote")
    ap.add_argument("--ckpt", type=int, default=-1)
    ap.add_argument("--log_root", type=str, default="logs")
    ap.add_argument("--out", type=str, default="/content/eval.mp4")
    ap.add_argument("--grid", type=str, default="/content/grid_floor.png")
    ap.add_argument("--steps_per_cmd", type=int, default=120)
    ap.add_argument("--fps", type=int, default=50)
    args = ap.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    log_dir = f"{args.log_root}/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {} # quiet eval

    env = Go2Env(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                 command_cfg=command_cfg, show_viewer=False, attach_camera=True,
                 camera_res=(960, 640), camera_pos=(2.4, 1.6, 1.1), camera_lookat=(0.3, 0.0, 0.25),
                 camera_fov=42, hiq=True, grid_texture=args.grid)

    ckpts = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
    if args.ckpt < 0:
        nums = sorted([int(f[6:-3]) for f in ckpts])
        ckpt = nums[-1]
    else:
        ckpt = args.ckpt

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)
    print("Loaded checkpoint", ckpt)

    # scripted command segments: (vx, vy, wz)
    segments = [(1.2, 0.0, 0.0), (0.6, 0.0, 1.0), (0.6, 0.0, -1.0), (1.5, 0.0, 0.0)]

    obs, _ = env.reset()
    env.camera.start_recording()

    with torch.no_grad():
        for (vx, vy, wz) in segments:
            env.commands[:, 0] = vx; env.commands[:, 1] = vy; env.commands[:, 2] = wz
            for i in range(args.steps_per_cmd):
                act = policy(obs)

                # --- BUG FIX: Unpack all 4 values returned by env.step() ---
                obs, rew, done, info = env.step(act)

                env.commands[:, 0] = vx; env.commands[:, 1] = vy; env.commands[:, 2] = wz # hold command

                # --- CAMERA TRACKING ADDITION ---
                # 1. Get the robot's base position (index 0 since num_envs=1)
                base_pos = env.robot.get_pos()[0].cpu().numpy()

                # 2. Maintain a trailing offset behind and slightly above the robot
                cam_pos = (base_pos[0] + 2.1, base_pos[1] + 1.6, 1.1)
                cam_lookat = (base_pos[0], base_pos[1], base_pos[2])

                # 3. Update the camera pose before rendering
                env.camera.set_pose(pos=cam_pos, lookat=cam_lookat)
                # --------------------------------

                env.camera.render()

    env.camera.stop_recording(save_to_filename=args.out, fps=args.fps)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
