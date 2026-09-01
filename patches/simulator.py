import os
import sys
import time
import traceback
from abc import abstractmethod

# gsplat JIT: the image defaults MAX_JOBS=10, which OOMs laptop RAM and
# SIGKILLs the process with no traceback. Pin Ada (RTX 4060 = sm_89) so
# nvcc does not compile every arch. Must run before importing gsplat.
os.environ["MAX_JOBS"] = os.environ.get("SUPERMARKET_GS_MAX_JOBS", "2")
os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get(
    "SUPERMARKET_TORCH_CUDA_ARCH_LIST", "8.9"
)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import glfw
from PIL import Image
import OpenGL.GL as gl

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from discoverse import DISCOVERSE_ASSETS_DIR
from discoverse.utils import BaseConfig, get_screen_scale

SIMULATION_WINDOW_TITLE = "Shentoon RobotStudio"

if sys.platform == "linux":
    try:
        import torch
        from gaussian_renderer.gs_renderer_mujoco import GSRendererMuJoCo
        DISCOVERSE_GAUSSIAN_RENDERER = True

    except ImportError:
        print("Warning: gaussian_splatting renderer not found. Please install the required packages to use it.")
        DISCOVERSE_GAUSSIAN_RENDERER = False
else:
    DISCOVERSE_GAUSSIAN_RENDERER = False

_SUPERMARKET_GS_DIRS = (
    "/workspace/supermarket_sorting_task/examples/supermarket_sorting/models/3dgs",
    "/workspace/supermarket_sorting_task/examples/supermarket_sorting/models/3dgs/shentoon",
)


def resolve_gs_model_path(path, name, assets_dir, hf_repo_id=None):
    """Prefer on-image supermarket PLYs before HuggingFace (which hangs offline)."""
    path = str(path)
    candidates = []
    if os.path.isabs(path):
        candidates.append(path)
    else:
        candidates.append(os.path.join(assets_dir, "3dgs", path))
        candidates.append(os.path.join(assets_dir, path))
        candidates.append(path)
    extra = os.environ.get("SUPERMARKET_GS_DIR", "").strip()
    search_dirs = list(_SUPERMARKET_GS_DIRS)
    if extra:
        search_dirs.insert(0, extra)
    base = os.path.basename(path)
    rel = path.replace("\\", "/")
    for folder in search_dirs:
        candidates.extend((
            os.path.join(folder, path),
            os.path.join(folder, rel),
            os.path.join(folder, base),
            os.path.join(folder, "shentoon", base),
        ))
    seen = set()
    for cand in candidates:
        cand = os.path.normpath(cand)
        if cand in seen:
            continue
        seen.add(cand)
        if os.path.isfile(cand):
            print(f"[gs] {name}: {cand}")
            return cand
    if hf_repo_id and not os.path.isabs(path):
        print(f"[gs] {name}: not on disk, HuggingFace {path} from {hf_repo_id}")
        from discoverse.utils.download_from_huggingface import download_from_huggingface
        return download_from_huggingface(path, hf_repo_id)
    print(f"[gs] {name}: FILE NOT FOUND ({path})")
    return path


def slim_gs_model_dict(gs_model_dict):
    """Keep robot + background + selected SKU Gaussians so an 8GB laptop can boot GS=1.

    Not true paging: GSRendererMuJoCo loads the whole dict at construct time.
    Change SUPERMARKET_GS_KINDS and restart the Server to test another batch.
    """
    kept = {}
    dropped = []
    no_background = os.environ.get("SUPERMARKET_GS_NO_BACKGROUND", "0") == "1"
    keep_obstacles = os.environ.get("SUPERMARKET_GS_KEEP_OBSTACLES", "0") == "1"
    raw_kinds = os.environ.get("SUPERMARKET_GS_KINDS", "kele").strip().lower()
    if raw_kinds in {"all", "*"}:
        kinds = None
        print("[gs] SUPERMARKET_GS_KINDS=all keeps every product ply (likely OOM on 8GB)", flush=True)
    else:
        kinds = [item.strip() for item in raw_kinds.replace(";", ",").split(",") if item.strip()]
        if not kinds:
            kinds = ["kele"]
        print(f"[gs] product batch: {kinds}", flush=True)
    for name, path in gs_model_dict.items():
        path_s = str(path).replace("\\", "/").lower()
        base = os.path.basename(path_s)
        stem = os.path.splitext(base)[0].replace("_fit", "")
        keep = (
            name == "background"
            or name.startswith(("agv", "slide", "head_", "lft_", "rgt_"))
            or "/mmk2/" in path_s
            or "/airbot_play/" in path_s
        )
        if kinds is None:
            keep = True
        elif any(kind == stem or kind in base for kind in kinds):
            keep = True
        if name == "background" and no_background:
            keep = False
        if keep_obstacles and "yellowbox" in base:
            keep = True
        if keep:
            kept[name] = path
        else:
            dropped.append(name)
    print(
        f"[gs] slim kept={len(kept)} dropped={len(dropped)} "
        f"(restart Server with a new SUPERMARKET_GS_KINDS to test another batch)",
        flush=True,
    )
    return kept


def should_slim_gs(total_memory_bytes=None):
    flag = os.environ.get("SUPERMARKET_GS_SLIM", "auto").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if flag in {"0", "false", "no", "off"}:
        return False
    if total_memory_bytes is None:
        return True
    return float(total_memory_bytes) < 12e9


def gs_should_skip_window(slim: bool) -> bool:
    """Skip a second GS pass for the GLFW free-camera on 8GB cards."""
    flag = os.environ.get("SUPERMARKET_GS_WINDOW", "auto").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return True
    if flag in {"1", "true", "yes", "on"}:
        return False
    return bool(slim)


def gs_filter_obs_cam_ids(obs_cam_ids, camera_names, slim: bool):
    """On 8GB, only GS-render the head camera; wrists stay MuJoCo."""
    ids = []
    for cam_id in obs_cam_ids:
        if cam_id not in ids:
            ids.append(cam_id)
    raw = os.environ.get("SUPERMARKET_GS_CAM_IDS", "auto").strip().lower()
    if raw not in {"", "auto"}:
        allowed = {
            int(item)
            for item in raw.replace(";", ",").split(",")
            if item.strip().lstrip("-").isdigit()
        }
        return [cam_id for cam_id in ids if cam_id in allowed]
    if not slim:
        return ids
    names = list(camera_names or [])
    head_ids = [
        cam_id
        for cam_id in ids
        if cam_id >= 0 and cam_id < len(names) and "head" in str(names[cam_id]).lower()
    ]
    if head_ids:
        return head_ids
    if 0 in ids:
        return [0]
    return ids[:1]


def setRenderOptions(options):
    options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = True
    options.frame = mujoco.mjtFrame.mjFRAME_BODY.value
    pass

class SimulatorBase:
    running = True
    obs = None
    img_rgb_obs_s = {}
    img_depth_obs_s = {}
    free_body_qpos_ids = {}

    cam_id = -1  # -1表示自由视角
    last_cam_id = -1
    render_cnt = 0
    camera_names = []
    camera_pose_changed = False
    camera_rmat = np.array([
        [ 0,  0, -1],
        [-1,  0,  0],
        [ 0,  1,  0],
    ])

    use_default_window_size = False
    mouse_pressed = {
        'left': False,
        'right': False,
        'middle': False
    }
    mouse_pos = {
        'x': 0,
        'y': 0
    }

    options = mujoco.MjvOption()

    def __init__(self, config:BaseConfig):
        self.config = config

        if self.config.mjcf_file_path.startswith("/"):
            self.mjcf_file = self.config.mjcf_file_path
        elif os.path.exists(self.config.mjcf_file_path):
            self.mjcf_file = self.config.mjcf_file_path
        else:
            self.mjcf_file = os.path.join(DISCOVERSE_ASSETS_DIR, self.config.mjcf_file_path)
        if os.path.exists(self.mjcf_file):
            print("mjcf found: {}".format(self.mjcf_file))
        else:
            print("\033[0;31;40mFailed to load mjcf: {}\033[0m".format(self.mjcf_file))
            raise FileNotFoundError("Failed to load mjcf: {}".format(self.mjcf_file))
        self.load_mjcf()
        self.decimation = self.config.decimation
        self.delta_t = self.mj_model.opt.timestep * self.decimation
        self.render_fps = self.config.render_set["fps"]

        if self.config.enable_render:
            self.free_camera = mujoco.MjvCamera()
            self.free_camera.fixedcamid = -1
            self.free_camera.type = mujoco._enums.mjtCamera.mjCAMERA_FREE
            mujoco.mjv_defaultFreeCamera(self.mj_model, self.free_camera)

            self.config.use_gaussian_renderer = self.config.use_gaussian_renderer and DISCOVERSE_GAUSSIAN_RENDERER
            if self.config.use_gaussian_renderer:
                print("[gs] resolving 3DGS ply files...")
                hf_repo_id = getattr(self.config, "hf_repo_id", "tatp/DISCOVERSE-models")
                allow_hf = os.environ.get("SUPERMARKET_GS_ALLOW_HF", "0") == "1"
                for name, path in list(self.config.gs_model_dict.items()):
                    self.config.gs_model_dict[name] = resolve_gs_model_path(
                        path,
                        name,
                        DISCOVERSE_ASSETS_DIR,
                        hf_repo_id if allow_hf else None,
                    )
                missing = [name for name, ply in self.config.gs_model_dict.items() if not os.path.isfile(str(ply))]
                if missing:
                    print(f"[gs] missing ply: {missing}")
                    print("[gs] set SUPERMARKET_GS_ALLOW_HF=1 only if you can reach HuggingFace")
                gpu_total = None
                try:
                    import torch
                    if torch.cuda.is_available():
                        props = torch.cuda.get_device_properties(0)
                        gpu_total = float(props.total_memory)
                        print(
                            f"[gs] gpu={props.name} total={gpu_total / 1e9:.1f}GB "
                            f"allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB",
                            flush=True,
                        )
                        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                except Exception as exc:
                    print(f"[gs] could not query CUDA: {exc}", flush=True)
                if should_slim_gs(gpu_total):
                    self.config.gs_model_dict = slim_gs_model_dict(self.config.gs_model_dict)
                    self.config.gs_render_sequential = True
                    self._gs_slim = True
                else:
                    self._gs_slim = False
                print("[gs] constructing GSRendererMuJoCo (slow, uses VRAM)...", flush=True)
                print("[gs] if the process exits here with no traceback, the GPU OOM-killer SIGKILL'd it", flush=True)
                started = time.time()
                try:
                    self.gs_renderer = GSRendererMuJoCo(self.config.gs_model_dict, self.mj_model)
                except Exception:
                    print("[gs] GSRendererMuJoCo failed. If this is CUDA OOM, stop the Client first (YOLO+GS share the 4060).")
                    traceback.print_exc()
                    raise
                self.last_cam_id = self.cam_id
                self.show_gaussian_img = True
                print(f"[gs] renderer ready in {time.time() - started:.1f}s", flush=True)
                print(
                    f"[gs] first render will JIT-compile gsplat "
                    f"(MAX_JOBS={os.environ.get('MAX_JOBS')} "
                    f"TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST')}). "
                    f"Wait for '[gs] frame0'; do not Ctrl+C.",
                    flush=True,
                )
                print(
                    f"[gs] slim={getattr(self, '_gs_slim', False)} "
                    f"sequential={getattr(self.config, 'gs_render_sequential', False)} "
                    f"skip_window_gs={gs_should_skip_window(getattr(self, '_gs_slim', False))}",
                    flush=True,
                )
                try:
                    import torch
                    if torch.cuda.is_available():
                        print(
                            f"[gs] after load allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB "
                            f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB",
                            flush=True,
                        )
                except Exception:
                    pass

        self.window = None
        self.glfw_initialized = False
        
        if not hasattr(self.config.render_set, "window_title"):
            self.config.render_set["window_title"] = SIMULATION_WINDOW_TITLE
        
        if self.config.enable_render and not self.config.headless:
            try:
                if not glfw.init():
                    raise RuntimeError("无法初始化GLFW")
                self.glfw_initialized = True
                
                # 设置OpenGL版本和窗口属性
                glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
                glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
                glfw.window_hint(glfw.VISIBLE, True)

                # 如果设置了use_default_window_size，禁用窗口最大化功能
                if self.use_default_window_size:
                    # 禁用窗口最大化
                    glfw.window_hint(glfw.MAXIMIZED, False)
                    # 确保窗口有装饰（标题栏等）
                    glfw.window_hint(glfw.DECORATED, True)
                    # 允许用户手动调整窗口大小
                    glfw.window_hint(glfw.RESIZABLE, True)
                    print("已禁用窗口最大化功能，但允许调整窗口大小")

                # 创建窗口
                self.window = glfw.create_window(
                    self.config.render_set["width"],
                    self.config.render_set["height"],
                    self.config.render_set.get("window_title", SIMULATION_WINDOW_TITLE),
                    None, None
                )
                
                if not self.window:
                    glfw.terminate()
                    raise RuntimeError("无法创建GLFW窗口")
                
                glfw.make_context_current(self.window)
                glfw.swap_interval(1)

                # 设置窗口最大尺寸
                glfw.set_window_size_limits(self.window, 320, 240, self.mj_model.vis.global_.offwidth, self.mj_model.vis.global_.offheight)

                # 初始化OpenGL设置
                gl.glClearColor(0.0, 0.0, 0.0, 1.0)
                gl.glShadeModel(gl.GL_SMOOTH)
                gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
                
                # 设置回调
                glfw.set_key_callback(self.window, self.on_key)
                glfw.set_cursor_pos_callback(self.window, self.on_mouse_move)
                glfw.set_mouse_button_callback(self.window, self.on_mouse_button)
                glfw.set_scroll_callback(self.window, self.on_mouse_scroll)
                
                # 如果设置了use_default_window_size，添加窗口大小变化回调
                if self.use_default_window_size:
                    glfw.set_window_maximize_callback(self.window, self.maximize_callback)

                if sys.platform == "darwin":
                    self.screen_scale = get_screen_scale(0)
                    gl.glPixelZoom(self.screen_scale, self.screen_scale)
                else:
                    self.screen_scale = 1

                # 注册清理函数
                import atexit
                atexit.register(self._cleanup_before_exit)

            except Exception as e:
                print(f"GLFW初始化失败: {e}")
                if self.glfw_initialized:
                    glfw.terminate()
                self.config.headless = True
                self.window = None

        self.last_render_time = time.time()
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def maximize_callback(self, window, maximized):
        if self.use_default_window_size and maximized:
            glfw.restore_window(window)

    def object_pose(self, body_name):
        """获取物体的位姿（位置xyz和朝向wxyz）"""
        try:
            qid = self.mj_model.jnt_qposadr[self.free_body_qpos_ids[body_name]]
            return self.mj_data.qpos[qid:qid+7][...]
        except KeyError:
            raise KeyError(f"Body name '{body_name}' not found in free_body_qpos_ids. Available bodies: {list(self.free_body_qpos_ids.keys())}")
    
    def get_joint_position(self, joint_name):
        return self.mj_data.qpos[self.mj_model.joint(joint_name).qposadr]
    
    def set_joint_position(self, joint_name, value):
        self.mj_data.qpos[self.mj_model.joint(joint_name).qposadr] = value

    def load_mjcf(self):
        if self.mjcf_file.endswith(".xml"):
            self.mj_model = mujoco.MjModel.from_xml_path(self.mjcf_file)
        elif self.mjcf_file.endswith(".mjb"):
            self.mj_model = mujoco.MjModel.from_binary_path(self.mjcf_file)
        self.mj_model.opt.timestep = self.config.timestep
        # self.mj_model.vis.quality.shadowsize = 4096 * 8
        self.mj_data = mujoco.MjData(self.mj_model)
        if self.config.enable_render:
            for i in range(self.mj_model.ncam):
                self.camera_names.append(self.mj_model.camera(i).name)

            if type(self.config.obs_rgb_cam_id) is int:
                assert -2 < self.config.obs_rgb_cam_id < len(self.camera_names), "Invalid obs_rgb_cam_id {}".format(self.config.obs_rgb_cam_id)
                tmp_id = self.config.obs_rgb_cam_id
                self.config.obs_rgb_cam_id = [tmp_id]
            elif type(self.config.obs_rgb_cam_id) is list:
                for cam_id in self.config.obs_rgb_cam_id:
                    assert -2 < cam_id < len(self.camera_names), "Invalid obs_rgb_cam_id {}".format(cam_id)
            elif self.config.obs_rgb_cam_id is None:
                self.config.obs_rgb_cam_id = []
            
            if type(self.config.obs_depth_cam_id) is int:
                assert -2 < self.config.obs_depth_cam_id < len(self.camera_names), "Invalid obs_depth_cam_id {}".format(self.config.obs_depth_cam_id)
            elif type(self.config.obs_depth_cam_id) is list:
                for cam_id in self.config.obs_depth_cam_id:
                    assert -2 < cam_id < len(self.camera_names), "Invalid obs_depth_cam_id {}".format(cam_id)
            elif self.config.obs_depth_cam_id is None:
                self.config.obs_depth_cam_id = []
        
            try:
                import screeninfo
                monitors = screeninfo.get_monitors()
                screen_width, screen_height = 1920, 1080
                for m in monitors:
                    if getattr(m, "is_primary", False) or getattr(m, "isPrimary", False):
                        screen_width, screen_height = m.width, m.height
                        break
                else:
                    if monitors:
                        screen_width, screen_height = monitors[0].width, monitors[0].height
            except Exception as e:
                screen_width, screen_height = 1920, 1080
                print(f"screeninfo error: {e}, using default screen size: {screen_width}x{screen_height}")
                self.use_default_window_size = True

            self.mj_model.vis.global_.offwidth = max(self.mj_model.vis.global_.offwidth, screen_width)
            self.mj_model.vis.global_.offheight = max(self.mj_model.vis.global_.offheight, screen_height)
            self.renderer = mujoco.Renderer(self.mj_model, self.config.render_set["height"], self.config.render_set["width"])

        for i in range(self.mj_model.nbody):
            if len(self.mj_model.body(i).name) and self.mj_model.body(i).dofnum == 6:
                jq_id = np.where(self.mj_model.jnt_bodyid == self.mj_model.body(i).id)[0]
                if jq_id.size:
                    self.free_body_qpos_ids[self.mj_model.body(i).name] = int(jq_id[0])

        self.post_load_mjcf()

    def post_load_mjcf(self):
        pass

    def update_renderer_window_size(self, width, height):
        self.renderer._width = width
        self.renderer._height = height
        self.renderer._rect.width = width
        self.renderer._rect.height = height

    def get_current_window_size(self):
        width = int(self.config.render_set["width"])
        height = int(self.config.render_set["height"])

        if not self.config.headless and self.window is not None:
            fb_width, fb_height = glfw.get_framebuffer_size(self.window)
            if fb_width > 0 and fb_height > 0:
                screen_scale = getattr(self, "screen_scale", 1)
                width = max(1, int(fb_width / screen_scale))
                height = max(1, int(fb_height / screen_scale))

        return width, height

    def _convert_gs_render_results(self, results_tensor):
        results = {}
        for cid, (rgb_tensor, depth_tensor) in results_tensor.items():
            rgb = (255. * torch.clamp(rgb_tensor, 0.0, 1.0)).to(torch.uint8).cpu().numpy()
            depth = depth_tensor.cpu().numpy()
            results[cid] = (rgb, depth)
        return results

    def update_texture(self, texture_name, mtl_img_pil, no_render=False):
        """更新纹理"""
        if not hasattr(self, 'renderer') or self.renderer is None:
            print(f"Renderer not initialized, cannot update texture: {texture_name}")
            return False

        try:
            tex_id = self.renderer.model.tex(texture_name).id
        except Exception as e:
            print(f"Texture '{texture_name}' not found: {e}")
            return False
        
        if not no_render:
            self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
            self.renderer.render()
        
        tex_bind_id = self.renderer._mjr_context.texture[tex_id]
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_bind_id)
        
        try:
            width = gl.glGetTexLevelParameteriv(gl.GL_TEXTURE_2D, 0, gl.GL_TEXTURE_WIDTH)
            height = gl.glGetTexLevelParameteriv(gl.GL_TEXTURE_2D, 0, gl.GL_TEXTURE_HEIGHT)
        except Exception as e:
            print(f"Error getting texture dimensions: {e}")
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            return False

        try:
            if mtl_img_pil.mode != 'RGB':
                mtl_img_pil = mtl_img_pil.convert('RGB')

            if mtl_img_pil.size != (width, height):
                mtl_img_pil = mtl_img_pil.resize((width, height), Image.Resampling.LANCZOS)
            
            mtl_img = np.array(mtl_img_pil)
            mtl_img = np.flipud(mtl_img)
            mtl_img = np.ascontiguousarray(mtl_img, dtype=np.uint8)

            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)

            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, width, height, 
                              gl.GL_RGB, gl.GL_UNSIGNED_BYTE, mtl_img.tobytes())
            
        except Exception as e:
            print(f"Error processing image for texture '{texture_name}': {e}")
            return False
        finally:
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            
        return True

    def render(self):
        self.render_cnt += 1

        render_width = int(self.config.render_set["width"])
        render_height = int(self.config.render_set["height"])
        window_width, window_height = self.get_current_window_size()
        display_result = None

        self.update_renderer_window_size(render_width, render_height)
        use_gs = (
            getattr(self, "gs_renderer", None) is not None
            and bool(self.config.use_gaussian_renderer)
            and bool(getattr(self, "show_gaussian_img", False))
        )
        if use_gs:
            try:
                slim = bool(getattr(self, "_gs_slim", False))
                obs_cam_ids = gs_filter_obs_cam_ids(
                    list(self.config.obs_rgb_cam_id) + list(self.config.obs_depth_cam_id),
                    self.camera_names,
                    slim,
                )
                skip_window_gs = gs_should_skip_window(slim)
                display_cam_id = None
                if (
                    not skip_window_gs
                    and not self.config.headless
                    and self.window is not None
                ):
                    display_cam_id = self.cam_id
                render_cam_ids = obs_cam_ids.copy()
                if display_cam_id is not None and (window_width, window_height) == (render_width, render_height):
                    if display_cam_id not in render_cam_ids:
                        render_cam_ids.append(display_cam_id)

                if len(render_cam_ids) > 0 or display_cam_id is not None:
                    if -1 in render_cam_ids or display_cam_id == -1:
                        self.renderer.update_scene(self.mj_data, self.free_camera, self.options)

                    self.gs_renderer.update_gaussians(self.mj_data)
                    self.batch_render_results = {}

                    if len(render_cam_ids) > 0:
                        if getattr(self.config, "gs_render_sequential", False) or slim:
                            for render_cam_id in render_cam_ids:
                                results_tensor = self.gs_renderer.render(
                                    self.mj_model,
                                    self.mj_data,
                                    [render_cam_id],
                                    render_width,
                                    render_height,
                                    self.free_camera
                                )
                                self.batch_render_results.update(
                                    self._convert_gs_render_results(results_tensor)
                                )
                        else:
                            results_tensor = self.gs_renderer.render(
                                self.mj_model,
                                self.mj_data,
                                render_cam_ids,
                                render_width,
                                render_height,
                                self.free_camera
                            )
                            self.batch_render_results = self._convert_gs_render_results(results_tensor)

                    if display_cam_id is not None:
                        if (window_width, window_height) == (render_width, render_height) and display_cam_id in self.batch_render_results:
                            display_result = self.batch_render_results[display_cam_id]
                        else:
                            window_results_tensor = self.gs_renderer.render(
                                self.mj_model,
                                self.mj_data,
                                [display_cam_id],
                                window_width,
                                window_height,
                                self.free_camera
                            )
                            display_result = self._convert_gs_render_results(window_results_tensor).get(display_cam_id)

                    for cid, (rgb, depth) in self.batch_render_results.items():
                        self.img_rgb_obs_s[cid] = rgb
                        self.img_depth_obs_s[cid] = depth

                    # Wrists (and any cam not in the GS set) stay MuJoCo so P3
                    # can keep a single head GS view without 4x VRAM.
                    for cam_id in self.config.obs_rgb_cam_id:
                        if cam_id not in self.batch_render_results:
                            self.img_rgb_obs_s[cam_id] = self.getRgbImg(cam_id)
                    for cam_id in self.config.obs_depth_cam_id:
                        if cam_id not in self.batch_render_results:
                            self.img_depth_obs_s[cam_id] = self.getDepthImg(cam_id)

                    if not getattr(self, "_gs_frame_logged", False):
                        print(
                            f"[gs] frame0 cam_id={self.cam_id} obs={self.config.obs_rgb_cam_id} "
                            f"gs_cams={render_cam_ids} n_gs_imgs={len(self.batch_render_results)} "
                            f"display={'ok' if display_result is not None else 'mujoco'}",
                            flush=True,
                        )
                        self._gs_frame_logged = True
                    if slim and self.render_cnt % 90 == 1:
                        try:
                            import torch
                            if torch.cuda.is_available():
                                free, total = torch.cuda.mem_get_info(0)
                                print(
                                    f"[gs] vram frame={self.render_cnt} "
                                    f"alloc={torch.cuda.memory_allocated() / 1e9:.2f}GB "
                                    f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB "
                                    f"free={free / 1e9:.2f}/{total / 1e9:.1f}GB",
                                    flush=True,
                                )
                        except Exception:
                            pass
            except Exception:
                if not getattr(self, "_gs_render_error_printed", False):
                    print("[gs] render failed, falling back to MuJoCo primitives", flush=True)
                    traceback.print_exc()
                    self._gs_render_error_printed = True
                display_result = None
                use_gs = False

        if not use_gs:
            depth_rendering = self.renderer._depth_rendering
            self.renderer.disable_depth_rendering()
            for id in self.config.obs_rgb_cam_id:
                img = self.getRgbImg(id)
                self.img_rgb_obs_s[id] = img

            self.renderer.enable_depth_rendering()
            for id in self.config.obs_depth_cam_id:
                img = self.getDepthImg(id)
                self.img_depth_obs_s[id] = img
            self.renderer._depth_rendering = depth_rendering
        
        if not self.config.headless and self.window is not None:
            self.update_renderer_window_size(window_width, window_height)
            if not self.renderer._depth_rendering:
                if self.config.use_gaussian_renderer and self.show_gaussian_img and display_result is not None:
                    img_vis = display_result[0]
                elif self.cam_id in self.config.obs_rgb_cam_id and (window_width, window_height) == (render_width, render_height):
                    img_vis = self.img_rgb_obs_s[self.cam_id]
                else:
                    img_vis = self.getRgbImg(self.cam_id)
            else:
                if self.config.use_gaussian_renderer and self.show_gaussian_img and display_result is not None:
                    img_depth = display_result[1]
                elif self.cam_id in self.config.obs_depth_cam_id and (window_width, window_height) == (render_width, render_height):
                    img_depth = self.img_depth_obs_s[self.cam_id]
                else:
                    img_depth = self.getDepthImg(self.cam_id)
                
                if img_depth is not None:
                    img_vis = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=255./self.config.max_render_depth), cv2.COLORMAP_JET)
                else:
                    img_vis = None

            try:
                if glfw.window_should_close(self.window):
                    self.running = False
                    return
                    
                glfw.make_context_current(self.window)
                fb_width, fb_height = glfw.get_framebuffer_size(self.window)
                gl.glViewport(0, 0, fb_width, fb_height)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)

                if img_vis is not None:
                    img_vis = img_vis[::-1]
                    img_vis = np.ascontiguousarray(img_vis)
                    gl.glWindowPos2i(0, 0)
                    gl.glDrawPixels(img_vis.shape[1], img_vis.shape[0], gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img_vis.tobytes())
                
                glfw.swap_buffers(self.window)
                glfw.poll_events()
                
                if self.config.sync:
                    current_time = time.time()
                    wait_time = max(1.0/self.render_fps - (current_time - self.last_render_time), 0)
                    if wait_time > 0:
                        time.sleep(wait_time)
                    self.last_render_time = time.time()
                    
            except Exception as e:
                print(f"渲染错误: {e}")

    def getRgbImg(self, cam_id):
        if cam_id == -1:
            self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
        elif cam_id > -1:
            self.renderer.update_scene(self.mj_data, self.camera_names[cam_id], self.options)
        else:
            return None
        rgb_img = self.renderer.render()
        return rgb_img

    def getDepthImg(self, cam_id):
        if cam_id == -1:
            self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
        elif cam_id > -1:
            self.renderer.update_scene(self.mj_data, self.camera_names[cam_id], self.options)
        else:
            return None
        depth_img = self.renderer.render()
        return depth_img

    def on_mouse_move(self, window, xpos, ypos):
        if self.cam_id == -1:
            dx = xpos - self.mouse_pos['x']
            dy = ypos - self.mouse_pos['y']
            _, height = self.get_current_window_size()
            
            action = None
            if self.mouse_pressed['left']:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
            elif self.mouse_pressed['right']:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_V
            elif self.mouse_pressed['middle']:
                action = mujoco.mjtMouse.mjMOUSE_ZOOM

            if action is not None:
                self.camera_pose_changed = True
                mujoco.mjv_moveCamera(self.mj_model,  action,  dx/height,  dy/height, self.renderer.scene, self.free_camera)

        self.mouse_pos['x'] = xpos
        self.mouse_pos['y'] = ypos

    def on_mouse_button(self, window, button, action, mods):
        is_pressed = action == glfw.PRESS
        
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_pressed['left'] = is_pressed
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.mouse_pressed['right'] = is_pressed
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.mouse_pressed['middle'] = is_pressed

    def on_mouse_scroll(self, window, xoffset, yoffset):
        self.free_camera.distance -= yoffset * 0.1
        if self.free_camera.distance < 0.1:
            self.free_camera.distance = 0.1

    def on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            is_ctrl_pressed = (mods & glfw.MOD_CONTROL)
            
            if is_ctrl_pressed:
                if key == glfw.KEY_G:  # Ctrl + G
                    if self.config.use_gaussian_renderer:
                        self.show_gaussian_img = not self.show_gaussian_img
                        self.gs_renderer.need_rerender = True
                        print(f"[gs] show_gaussian_img={self.show_gaussian_img}", flush=True)
                elif key == glfw.KEY_D:  # Ctrl + D
                    if self.config.use_gaussian_renderer:
                        self.gs_renderer.need_rerender = True
                    if self.renderer._depth_rendering:
                        self.renderer.disable_depth_rendering()
                    else:
                        self.renderer.enable_depth_rendering()
            else:
                if key == glfw.KEY_H:  # 'h': 显示帮助
                    self.printHelp()
                elif key == glfw.KEY_P:  # 'p': 打印信息
                    self.printMessage()
                elif key == glfw.KEY_G:  # 'g': 切换高斯（帮助文案写的是 G）
                    if getattr(self, "gs_renderer", None) is not None:
                        self.show_gaussian_img = not self.show_gaussian_img
                        self.gs_renderer.need_rerender = True
                        print(f"[gs] show_gaussian_img={self.show_gaussian_img}", flush=True)
                elif key == glfw.KEY_R:  # 'r': 重置状态
                    self.reset()
                elif key == glfw.KEY_ESCAPE:  # ESC: 切换到自由视角
                    self.cam_id = -1
                    self.camera_pose_changed = True
                elif key == glfw.KEY_RIGHT_BRACKET:  # ']': 下一个相机
                    if self.mj_model.ncam:
                        self.cam_id += 1
                        self.cam_id = self.cam_id % self.mj_model.ncam
                elif key == glfw.KEY_LEFT_BRACKET:  # '[': 上一个相机
                    if self.mj_model.ncam:
                        self.cam_id += self.mj_model.ncam - 1
                        self.cam_id = self.cam_id % self.mj_model.ncam

    def printHelp(self):
        """打印帮助信息"""
        print("\n=== 键盘控制说明 ===")
        print("H: 显示此帮助信息")
        print("P: 打印当前状态信息")
        print("R: 重置模拟器状态")
        print("G: 切换高斯渲染（如果可用）")
        print("D: 切换深度渲染")
        print("Ctrl+G: 组合键切换高斯模式")
        print("Ctrl+D: 组合键切换深度图模式")
        print("ESC: 切换到自由视角")
        print("[: 切换到上一个相机")
        print("]: 切换到下一个相机")
        print("\n=== 鼠标控制说明 ===")
        print("左键拖动: 旋转视角")
        print("右键拖动: 平移视角")
        print("中键拖动: 缩放视角")
        print("================\n")

    def printMessage(self):
        """打印当前状态信息"""
        print("\n=== 当前状态 ===")
        print(f"当前相机ID: {self.cam_id}")
        if self.cam_id >= 0:
            print(f"相机名称: {self.camera_names[self.cam_id]}")
        print(f"高斯渲染: {'开启' if self.show_gaussian_img else '关闭'}")
        print(f"深度渲染: {'开启' if self.renderer._depth_rendering else '关闭'}")
        print("==============\n")

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.camera_pose_changed = True

    def getCameraPose(self, cam_id):
        if cam_id == -1:
            rotation_matrix = self.camera_rmat @ Rotation.from_euler('xyz', [self.free_camera.elevation * np.pi / 180.0, self.free_camera.azimuth * np.pi / 180.0, 0.0]).as_matrix()
            camera_position = self.free_camera.lookat + self.free_camera.distance * rotation_matrix[:3,2]
        else:
            rotation_matrix = np.array(self.mj_data.camera(self.camera_names[cam_id]).xmat).reshape((3,3))
            camera_position = self.mj_data.camera(self.camera_names[cam_id]).xpos

        return camera_position, Rotation.from_matrix(rotation_matrix).as_quat()[[3,0,1,2]]

    def _cleanup_before_exit(self):
        """在Python退出前执行的清理函数"""
        try:
            # 如果GLFW上下文有效，先清理Mujoco渲染器
            if hasattr(self, 'renderer'):
                try:
                    del self.renderer
                except Exception:
                    pass

            # 然后清理GLFW资源
            if hasattr(self, 'window') and self.window is not None:
                try:
                    glfw.destroy_window(self.window)
                except Exception:
                    pass
                self.window = None
            
            # 最后终止GLFW
            if hasattr(self, 'glfw_initialized') and self.glfw_initialized:
                try:
                    glfw.terminate()
                except Exception:
                    pass
                self.glfw_initialized = False
            
        except Exception:
            pass

    # ------------------------------------------------------------------------------
    # ---------------------------------- Override ----------------------------------
    def reset(self):
        self.resetState()
        if self.config.enable_render:
            self.render()
        self.render_cnt = 0
        return self.getObservation()

    def updateControl(self, action):
        pass

    # 包含了一些需要子类实现的抽象方法
    @abstractmethod
    def post_physics_step(self):
        pass

    @abstractmethod
    def getChangedObjectPose(self):
        raise NotImplementedError("pubObjectPose is not implemented")

    @abstractmethod
    def checkTerminated(self):
        raise NotImplementedError("checkTerminated is not implemented")    

    @abstractmethod
    def getObservation(self):
        raise NotImplementedError("getObservation is not implemented")

    @abstractmethod
    def getPrivilegedObservation(self):
        raise NotImplementedError("getPrivilegedObservation is not implemented")

    @abstractmethod
    def getReward(self):
        raise NotImplementedError("getReward is not implemented")
    
    # ---------------------------------- Override ----------------------------------
    # ------------------------------------------------------------------------------

    def step(self, action=None): # 主要的仿真步进函数
        for _ in range(self.decimation):
            self.updateControl(action)
            mujoco.mj_step(self.mj_model, self.mj_data)

        terminated = self.checkTerminated()
        if terminated:
            self.resetState()
        
        self.post_physics_step()
        if self.config.enable_render and self.render_cnt-1 < self.mj_data.time * self.render_fps:
            self.render()

        return self.getObservation(), self.getPrivilegedObservation(), self.getReward(), terminated, {}

    def view(self):
        self.mj_data.time += self.delta_t
        self.mj_data.qvel[:] = 0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        if self.render_cnt-1 < self.mj_data.time * self.render_fps:
            self.render()
