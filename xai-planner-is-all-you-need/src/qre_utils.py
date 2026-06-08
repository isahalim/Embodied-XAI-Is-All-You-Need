import os, json, re, math
import numpy as np

# ----------------------------------------------------------------------------------
# Bucketed action maps (LOCKED, see plan §5.7). Single integer token => clean saliency anchor.
TURN_TO_WZ  = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}   # 0=hard-right ... 4=hard-left
SPEED_TO_VX = {0: 0.0, 1: 0.6, 2: 1.2}                       # 0=stop, 1=walk, 2=fast
VY_FIXED    = 0.0

def bucket_to_command(turn, speed):
    """Map VLM buckets to a clamped (v_x, v_y, w_z) command."""
    turn  = int(max(0, min(4, turn)))
    speed = int(max(0, min(2, speed)))
    return float(SPEED_TO_VX[speed]), float(VY_FIXED), float(TURN_TO_WZ[turn])

# ----------------------------------------------------------------------------------
# High-quality render profile (plan §5.8)
def make_grid_texture(path, size_px=1024, n_cells=16, line_px=4):
    """Black floor with thin white grid lines. Saved as PNG, applied as a diffuse texture."""
    import imageio.v2 as imageio
    img = np.zeros((size_px, size_px, 3), dtype=np.uint8)  # black
    step = size_px // n_cells
    for i in range(0, size_px + 1, step):
        lo = max(0, i - line_px // 2); hi = min(size_px, i + line_px // 2 + 1)
        img[lo:hi, :, :] = 255
        img[:, lo:hi, :] = 255
    imageio.imwrite(path, img)
    return path

def build_hiq_scene(gs, dt=0.02, rendered_envs_idx=(0,)):
    """Construct a gs.Scene with lights + shadows when supported, else a safe fallback.
    Returns (scene, used_hiq: bool). VERIFY in NB0 — VisOptions fields vary by version."""
    rigid = gs.options.RigidOptions(dt=dt, constraint_solver=gs.constraint_solver.Newton,
                                    enable_collision=True, enable_joint_limit=True)
    sim = gs.options.SimOptions(dt=dt, substeps=2)
    # Try a rich VisOptions; fall back progressively.
    vis_attempts = [
        dict(rendered_envs_idx=list(rendered_envs_idx), shadow=True,
             ambient_light=(0.25, 0.25, 0.25),
             lights=[{"type": "directional", "dir": (-0.5, -0.6, -1.0),
                      "color": (1.0, 1.0, 1.0), "intensity": 6.0}]),
        dict(rendered_envs_idx=list(rendered_envs_idx), shadow=True),
        dict(rendered_envs_idx=list(rendered_envs_idx)),
    ]
    for i, kw in enumerate(vis_attempts):
        try:
            scene = gs.Scene(sim_options=sim, vis_options=gs.options.VisOptions(**kw),
                             rigid_options=rigid, show_viewer=False)
            return scene, (i == 0)
        except TypeError:
            continue
    scene = gs.Scene(sim_options=sim, rigid_options=rigid, show_viewer=False)
    return scene, False

def add_ground(gs, scene, grid_texture_path=None):
    """Add the floor. Try a grid-textured surface, else a plain dark plane. VERIFY in NB0."""
    morph = gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
    if grid_texture_path and os.path.exists(grid_texture_path):
        for surf in (
            lambda: gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=grid_texture_path)),
            lambda: gs.surfaces.Rough(diffuse_texture=gs.textures.ImageTexture(image_path=grid_texture_path)),
        ):
            try:
                return scene.add_entity(morph, surface=surf())
            except Exception:
                continue
    try:
        return scene.add_entity(morph, surface=gs.surfaces.Default(color=(0.05, 0.05, 0.05)))
    except Exception:
        return scene.add_entity(morph)

def add_colored_box(gs, scene, pos, size, color):
    """Add a fixed colored rectangle (a candidate goal object). VERIFY surface API in NB0."""
    morph = gs.morphs.Box(pos=tuple(pos), size=tuple(size), fixed=True)
    for surf in (lambda: gs.surfaces.Default(color=tuple(color)),
                 lambda: gs.surfaces.Rough(color=tuple(color))):
        try:
            return scene.add_entity(morph, surface=surf())
        except Exception:
            continue
    return scene.add_entity(morph)

# ----------------------------------------------------------------------------------
# Camera geometry for saliency (plan §5.4)
def camera_intrinsics(res, fov_deg):
    """Pinhole K from horizontal-ish fov + resolution. Genesis fov is vertical; adjust if NB0 shows otherwise."""
    w, h = res
    fov = math.radians(fov_deg)
    fy = (h / 2) / math.tan(fov / 2)
    fx = fy  # square pixels
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K

def look_at_extrinsics(cam_pos, lookat, up=(0, 0, 1)):
    """World->camera transform T_cw (4x4) and camera->world T_wc. OpenCV convention (+z forward)."""
    cam_pos = np.asarray(cam_pos, float); lookat = np.asarray(lookat, float); up = np.asarray(up, float)
    f = lookat - cam_pos; f /= (np.linalg.norm(f) + 1e-9)
    r = np.cross(f, up); r /= (np.linalg.norm(r) + 1e-9)
    u = np.cross(r, f)
    R_wc = np.stack([r, -u, f], axis=1)         # camera axes in world (x right, y down, z forward)
    T_wc = np.eye(4); T_wc[:3, :3] = R_wc; T_wc[:3, 3] = cam_pos
    T_cw = np.linalg.inv(T_wc)
    return T_cw, T_wc

def unproject(u, v, depth, K, T_wc):
    """Pixel (u,v) + metric depth -> world XYZ. depth is along the camera +z axis."""
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    p_cam = np.array([x, y, depth, 1.0])
    return (T_wc @ p_cam)[:3]

# ----------------------------------------------------------------------------------
# Render-and-assert sanity gate (plan §5.2)
def render_camera(camera, want_depth=True):
    """Call camera.render and return (rgb_uint8, depth_or_None). VERIFY signature in NB0.
    Genesis camera.render typically returns (rgb, depth, seg, normal) for requested buffers."""
    out = None
    for kwargs in ({"rgb": True, "depth": want_depth}, {}):
        try:
            out = camera.render(**kwargs); break
        except TypeError:
            continue
    if not isinstance(out, (tuple, list)):
        rgb = np.asarray(out); return rgb, None
    rgb = np.asarray(out[0])
    depth = None
    if want_depth and len(out) > 1 and out[1] is not None:
        depth = np.asarray(out[1])
    return rgb, depth

def sanity_gate(scene, robot, camera, save_png, n_steps=8, bound=5.0, z_max=2.0):
    """Step with tiny random actions, assert physical sanity + a usable RGB(+depth) frame."""
    import torch, imageio.v2 as imageio
    for _ in range(n_steps):
        scene.step()
    pos = np.asarray(robot.get_pos().cpu()) if hasattr(robot.get_pos(), "cpu") else np.asarray(robot.get_pos())
    pos = pos.reshape(-1, 3)[0]
    assert np.all(np.isfinite(pos)), f"NaN/Inf base pos {pos}"
    assert abs(pos[0]) < bound and abs(pos[1]) < bound, f"Robot launched away: {pos}"
    assert 0 < pos[2] < z_max, f"Bad base height: {pos[2]}"
    rgb, depth = render_camera(camera, want_depth=True)
    assert rgb is not None and rgb.size > 0 and rgb.std() > 1.0, "Degenerate RGB frame"
    imageio.imwrite(save_png, rgb[..., :3].astype(np.uint8))
    depth_ok = depth is not None and np.isfinite(depth).any() and float(np.nanstd(depth)) > 1e-4
    return dict(base_pos=pos.tolist(), rgb_shape=tuple(rgb.shape),
                depth_present=depth is not None, depth_nondegenerate=bool(depth_ok),
                png=save_png)

# ----------------------------------------------------------------------------------
# Two-goal bias arena (plan §7). Red vs. blue, luminance-matched.
# sRGB relative luminance L = 0.2126 R + 0.7152 G + 0.0722 B. We pick a red and a blue with equal L.
def luminance(c):
    r, g, b = c
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

# Equal-luminance red/blue (L ~= 0.21): pure-ish red, brightened blue.
COLOR_RED  = (0.99, 0.05, 0.05)
COLOR_BLUE = (0.05, 0.05, 0.99)
# blue luminance is lower; scale blue up to match red's luminance via an added neutral component
def luminance_match(ref, other):
    """Return `other` blended toward white to match ref luminance (keeps hue dominant)."""
    Lref, Loth = luminance(ref), luminance(other)
    if Loth >= Lref:
        return other
    # add gray g to all channels so luminance rises to Lref
    g = (Lref - Loth)
    return tuple(min(1.0, c + g) for c in other)

def goal_positions(distance=3.0, lateral=1.2, left_color=None, right_color=None):
    """Two rectangles, equal forward distance, symmetric left/right. Robot starts at origin facing +x."""
    left  = dict(pos=(distance, +lateral, 0.25), size=(0.4, 0.4, 0.5), color=left_color,  side="left")
    right = dict(pos=(distance, -lateral, 0.25), size=(0.4, 0.4, 0.5), color=right_color, side="right")
    return left, right

# ----------------------------------------------------------------------------------
# Qwen2.5-VL: load, prompt contract, robust parse, single-token `turn` attention
QWEN_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

CONTRACT = (
    "You control a quadruped robot with an onboard camera. Look at the image and the instruction, "
    "then decide which way to steer. Respond with ONE JSON object and nothing else:\n"
    '{"reasoning": "<one short sentence>", "choice": "left|right", "turn": <0-4>, "speed": <0-2>}\n'
    "turn: 0=hard right, 1=right, 2=straight, 3=left, 4=hard left. "
    "speed: 0=stop, 1=walk, 2=fast. Output only the JSON."
)
CONTRACT_ACTION_ONLY = (
    "You control a quadruped robot with an onboard camera. Look at the image and the instruction, "
    "then decide which way to steer. Respond with ONE JSON object and nothing else:\n"
    '{"turn": <0-4>, "speed": <0-2>}\n'
    "turn: 0=hard right, 1=right, 2=straight, 3=left, 4=hard left. "
    "speed: 0=stop, 1=walk, 2=fast. Output only the JSON. Do not explain."
)

def load_qwen(device="cuda", dtype=None):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    if dtype is None: dtype = torch.bfloat16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_ID, torch_dtype=dtype, attn_implementation="eager", device_map=device)
    model.eval()
    processor = AutoProcessor.from_pretrained(QWEN_ID)
    return model, processor

def build_messages(pil_image, instruction, mode="reason"):
    contract = CONTRACT if mode == "reason" else CONTRACT_ACTION_ONLY
    return [{"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text", "text": f"{contract}\n\nInstruction: {instruction}"}]}]

def parse_action(raw):
    """Robustly extract {reasoning, choice, turn, speed} -> command. Returns dict incl. parse_status."""
    txt = raw.strip()
    txt = re.sub(r"^```(json)?|```$", "", txt, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    status = "ok"; obj = {}
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            status = "json_error"
    else:
        status = "no_json"
    def geti(k, default=2, lo=0, hi=4):
        try: return max(lo, min(hi, int(round(float(obj.get(k, default))))))
        except Exception: return default
    if status != "ok":
        # last-ditch: regex the integers
        t = re.search(r'turn"?\s*[:=]\s*(-?\d+)', txt); s = re.search(r'speed"?\s*[:=]\s*(-?\d+)', txt)
        if t and s:
            obj = {"turn": int(t.group(1)), "speed": int(s.group(1))}; status = "regex_recovered"
        else:
            return dict(parse_status="parse_failed", raw=raw, turn=2, speed=1,
                        choice=None, reasoning=None, v_x=0.6, v_y=0.0, w_z=0.0)
    turn = geti("turn", 2, 0, 4); speed = geti("speed", 1, 0, 2)
    vx, vy, wz = bucket_to_command(turn, speed)
    return dict(parse_status=status, raw=raw, turn=turn, speed=speed,
                choice=obj.get("choice"), reasoning=obj.get("reasoning"),
                v_x=vx, v_y=vy, w_z=wz)

def _vision_token_span(input_ids, model):
    """Index range of image tokens in the prompt (for attention slicing)."""
    import torch
    ids = input_ids[0].tolist()
    img_id = getattr(model.config, "image_token_id", None)
    if img_id is None:
        img_id = getattr(getattr(model, "config", None), "image_token_index", None)
    if img_id is None or img_id not in ids:
        return None
    idxs = [i for i, t in enumerate(ids) if t == img_id]
    return idxs[0], idxs[-1] + 1

def run_vlm(model, processor, pil_image, instruction, mode="reason",
            temperature=0.7, seed=0, max_new_tokens=96, want_attention=False):
    """Generate an action. If want_attention, also return the de-merged 2D attention over patches
    attributed to the single `turn` token. Returns a dict (see keys below)."""
    import torch
    from qwen_vl_utils import process_vision_info
    torch.manual_seed(seed)
    messages = build_messages(pil_image, instruction, mode=mode)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                      temperature=max(1e-4, temperature),
                      return_dict_in_generate=True, output_attentions=want_attention)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    seq = out.sequences[0]
    prompt_len = inputs.input_ids.shape[1]
    gen_ids = seq[prompt_len:]
    raw = processor.decode(gen_ids, skip_special_tokens=True)
    parsed = parse_action(raw)

    attn2d = None; grid_thw = None; turn_step = None
    if want_attention and getattr(out, "attentions", None) is not None:
        try:
            grid_thw = inputs["image_grid_thw"][0].tolist()  # [t, h, w] in patch units
            attn2d, turn_step = _turn_token_attention(model, processor, inputs, out, gen_ids, grid_thw)
        except Exception as e:
            parsed["attn_error"] = repr(e)
    parsed.update(dict(prompt_len=int(prompt_len), gen_text=raw,
                       attn2d=attn2d, grid_thw=grid_thw, turn_step=turn_step,
                       temperature=temperature, seed=seed, mode=mode))
    return parsed

def _turn_token_attention(model, processor, inputs, out, gen_ids, grid_thw):
    """Find the decode step whose token is the `turn` digit, aggregate last-layer attention
    (mean over heads) from that step to the vision tokens, de-merge via grid_thw (2x2 merge)."""
    import torch
    # locate the digit token following the literal 'turn'
    dec = [processor.tokenizer.decode([t]) for t in gen_ids.tolist()]
    turn_step = None
    for i, tok in enumerate(dec):
        if tok.strip().lstrip('-').isdigit():
            # heuristic: pick the digit that appears after the substring 'turn'
            ctx = "".join(dec[max(0, i - 6):i]).lower()
            if "turn" in ctx:
                turn_step = i; break
    if turn_step is None:
        for i, tok in enumerate(dec):  # fallback: first standalone digit
            if tok.strip().isdigit(): turn_step = i; break
    if turn_step is None:
        raise RuntimeError("no turn digit token found")
    span = _vision_token_span(inputs.input_ids, model)
    if span is None:
        raise RuntimeError("vision token span not found")
    v0, v1 = span
    # out.attentions: tuple over generated steps; each is tuple over layers (B, heads, q, kv)
    step_attn = out.attentions[turn_step][-1]            # last layer
    a = step_attn[0].mean(0)                              # mean over heads -> (q, kv)
    a = a[-1]                                             # query = the turn token (q-dim last)
    vis = a[v0:v1].float().cpu().numpy()                 # attention over vision tokens
    t, h, w = grid_thw
    merged_h, merged_w = h // 2, w // 2                  # 2x2 spatial token merge
    n = merged_h * merged_w
    vis = vis[:n] if vis.shape[0] >= n else np.pad(vis, (0, n - vis.shape[0]))
    grid = vis.reshape(merged_h, merged_w)
    grid = (grid - grid.min()) / (np.ptp(grid) + 1e-9)
    return grid, int(turn_step)
