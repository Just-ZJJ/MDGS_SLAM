import time

import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2,getWorld2View
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth

from collections import defaultdict
from utils.icp import IcpTracker
from utils.icp_utils import bilateralFilter_torch, compute_vertex_map, compute_normal_map, compute_confidence_map

from copy import deepcopy
from utils.general_utils import rot_compare,trans_compare

from utils.camera_utils import set_rays_od

def convert_poses(trajs):
    poses = []
    stamps = []
    for traj in trajs:
        stamp, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 = traj
        pose_ = np.eye(4)
        pose_[:3, :3] = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
        pose_[:3, 3] = np.array([t0, t1, t2])
        poses.append(pose_)
        stamps.append(stamp)
    return poses, stamps

class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.keyframe_trans_thes=config["Training"]["keyframe_trans_thes"]
        self.keyframe_theta_thes=config["Training"]["keyframe_theta_thes"]

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.opt_cameras=dict()
        self.device = "cuda:0"
        self.pause = False

        self.icp_tracker = IcpTracker(config)
        self.use_depth_filter = config["icp_params"]["depth_filter"]
        self.min_depth = config["icp_params"]["min_depth"]
        self.max_depth = config["icp_params"]["max_depth"]
        self.invalid_confidence_thresh = config["icp_params"]["invalid_confidence_thresh"]
        self.status = defaultdict(bool)
        self.use_orb_backend = config["orb_params"]["use_orb_backend"]
        self.global_optimize=config["orb_params"]["global_optimize"]
        self.orb_vocab_path = config["orb_params"]["orb_vocab_path"]
        self.orb_settings_path = config["Dataset"]["orb_settings_path"]
        self.orb_backend = None
        self.initialize_orb()

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def depth_filter(
            self,
            depth_map,
            K,
    ):

        if self.use_depth_filter:
            depth_map_filter = bilateralFilter_torch(depth_map, 5, 2, 2)
        else:
            depth_map_filter = depth_map
        valid_range_mask = (depth_map_filter > self.min_depth) & (depth_map_filter < self.max_depth)
        depth_map_filter[~valid_range_mask] = 0.0
        vertex_map_c = compute_vertex_map(depth_map_filter, K)
        normal_map_c = compute_normal_map(vertex_map_c)
        confidence_map = compute_confidence_map(normal_map_c, K)
        invalid_confidence_mask = ((normal_map_c == 0).all(dim=-1)) | (
                confidence_map < self.invalid_confidence_thresh
        )[..., 0]

        depth_map_filter[invalid_confidence_mask] = 0
        return depth_map_filter

    def refine_pose(self, pose_t1_t0, tracking_success,viewpoint):

        curr_image=(torch.clamp(torch.permute(torch.mul(viewpoint.original_image.cpu(), 255), (1, 2, 0)), 0, 255).to(torch.uint8)).numpy()
        curr_depth=np.array(viewpoint.depth *viewpoint.depth_scale).astype(np.uint16)
        timestamp=viewpoint.timestamp

        if tracking_success :
            #print("success")
            self.orb_backend.track_with_pose(
                curr_image,
                curr_depth,
                pose_t1_t0.astype(np.float32),
                timestamp,
            )
        else:
            self.orb_backend.track_with_orb_feature(
                curr_image,
                curr_depth,
                timestamp,
            )
        traj_history = self.orb_backend.get_trajectory_points()
        pose_es_t1, _ = convert_poses(traj_history[-2:])
        return pose_es_t1[-1]


    def initialize_orb(self):
        if self.use_orb_backend and self.orb_backend is None:
            import orbslam2
            print("init orb backend")
            self.orb_backend = orbslam2.System(
                self.orb_vocab_path, self.orb_settings_path, orbslam2.Sensor.RGBD
            )
            self.orb_backend.set_use_viewer(False)
            self.orb_backend.initialize(True)

    def initialize_orb_tracker(self, viewpoint):
        curr_image=(torch.clamp(torch.permute(torch.mul(viewpoint.original_image.cpu(), 255), (1, 2, 0)), 0, 255).to(torch.uint8)).numpy()
        curr_depth=np.array(viewpoint.depth *viewpoint.depth_scale).astype(np.uint16)
        timestamp=viewpoint.timestamp

        if self.use_orb_backend:
            self.orb_backend.process_image_rgbd(
                curr_image,
                curr_depth,
                timestamp,
            )
        self.status["initialized"] = True

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        curr_R=torch.eye(3)
        curr_T=torch.tensor([0,0,0])
        viewpoint.update_RT(curr_R, curr_T)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

        if not self.status["initialized"]:
            depth_map = torch.from_numpy(viewpoint.depth).to(dtype=torch.float32, device=self.device).unsqueeze(-1)
            depth_t1_filter=self.depth_filter(depth_map,viewpoint.K)
            self.icp_tracker.update_curr_status(depth_t1_filter, viewpoint.K)
            self.icp_tracker.move_last_status()
            self.initialize_orb_tracker(viewpoint)

    def tracking(self, cur_frame_idx, viewpoint):
        """"""
        frame_depth = torch.from_numpy(viewpoint.depth).to(dtype=torch.float32, device=self.device).unsqueeze(-1)
        frame_depth_filter=self.depth_filter(frame_depth,viewpoint.K)

        self.icp_tracker.update_curr_status(frame_depth_filter, viewpoint.K)
        pose_t1_t0, tracking_success = self.icp_tracker.predict_pose(viewpoint)

        if self.use_orb_backend:
            pose_t1_w = np.linalg.inv( self.refine_pose(pose_t1_t0, tracking_success,viewpoint) )
        else:
            prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
            pose_t0 = getWorld2View(prev.R.cpu().numpy(), prev.T.cpu().numpy())
            pose_t1_w = np.linalg.inv(pose_t1_t0) @ pose_t0
        curr_R=torch.tensor(pose_t1_w[:3,:3],dtype=viewpoint.R.dtype)
        curr_T=torch.tensor(pose_t1_w[:3,3],dtype=viewpoint.T.dtype)
        viewpoint.update_RT(curr_R,curr_T)
        self.icp_tracker.move_last_status()


        self.q_main2vis.put(
            gui_utils.GaussianPacket(
                current_frame=viewpoint,
                gtcolor=viewpoint.original_image,
                gtdepth=viewpoint.depth
                if not self.monocular
                else np.zeros((viewpoint.image_height, viewpoint.image_width)),
            )
        )
        
        render_pkg = render(
            viewpoint, self.gaussians, self.pipeline_params, self.background
        )
        image, depth, opacity = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["opacity"],
        )

        self.median_depth = get_median_depth(depth, opacity)

        return render_pkg


    def check_camera_motion(self, frame):
        # add keyframe
        keyframe_indx=self.kf_indices[-1]
        prev_rot = self.cameras[keyframe_indx].R.T
        prev_trans = self.cameras[keyframe_indx].T
        curr_rot = frame.R.T
        curr_trans = frame.T
        _, theta_diff = rot_compare(prev_rot, curr_rot)
        _, l2_diff = trans_compare(prev_trans, curr_trans)

        if theta_diff > self.keyframe_theta_thes or l2_diff > self.keyframe_trans_thes:
            return True
        else:
            return False

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        #return (point_ratio_2 < kf_overlap and dist_check2) or dist_check
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check or self.check_camera_motion(curr_frame)

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap, remove_idx):
        if self.use_orb_backend :
            new_poses, _ = convert_poses(self.orb_backend.get_trajectory_points())
            current_window_poses={}
            for idx in current_window :
                current_window_poses[idx]=np.linalg.inv(new_poses[idx])
            msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap, remove_idx, current_window_poses]
        else:
            msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap, remove_idx]

        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        #for kf_id, kf_R, kf_T in keyframes:
        #    self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset):

                    if self.save_results and self.use_orb_backend:
                        self.opt_cameras=deepcopy(self.cameras)
                        print("===== global optimize pose =====")
                        new_poses, _ = convert_poses(self.orb_backend.get_trajectory_points())
                        for i in range(len(self.cameras)) :
                            pose = np.linalg.inv(new_poses[i])
                            curr_R=torch.tensor(pose[:3,:3],dtype=torch.float32)
                            curr_T=torch.tensor(pose[:3,3],dtype=torch.float32)
                            self.opt_cameras[i].update_RT(curr_R,curr_T)
                        eval_ate(
                            self.opt_cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    elif self.save_results :
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                    save_gaussians(
                        self.gaussians, self.save_dir, "final", final=True
                    )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                render_pkg = self.tracking(cur_frame_idx, viewpoint)

                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf
                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map, removed
                    )
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
