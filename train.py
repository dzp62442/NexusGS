#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import json
import random
import shutil
import time
import numpy as np
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, l1_loss_mask, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, HFScene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from utils.sh_utils import eval_sh
from torchvision.utils import save_image


EVALUATION_FORMAT_VERSION = 2
ALL_18_VIEW_GROUP = "all_18_views"
NOVEL_12_VIEW_GROUP = "novel_12_views"
NOVEL_VIEW_COUNT = 12


def _atomic_write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary_path, path)


def _load_training_checkpoint(checkpoint_path):
    checkpoint_data = torch.load(checkpoint_path)
    if isinstance(checkpoint_data, dict):
        if checkpoint_data.get("format_version") != 2:
            raise ValueError(f"Unsupported training checkpoint: {checkpoint_path}")
        return checkpoint_data

    if not isinstance(checkpoint_data, (tuple, list)) or len(checkpoint_data) not in (2, 3):
        raise ValueError(f"Unsupported legacy training checkpoint: {checkpoint_path}")
    legacy = {
        "format_version": 1,
        "model_params": checkpoint_data[0],
        "iteration": checkpoint_data[1],
        "training_time_seconds": 0.0,
    }
    if len(checkpoint_data) == 3:
        legacy["training_time_seconds"] = checkpoint_data[2]
    return legacy

@torch.no_grad()
def clean_views(iteration, test_iterations, scene, gaussians, pipe, background):
    if iteration in test_iterations:
        visible_pnts = None
        for viewpoint_cam in scene.getTrainCameras().copy():
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            visibility_filter = render_pkg["visibility_filter"]
            if visible_pnts is None:
                visible_pnts = visibility_filter
            visible_pnts += visibility_filter
        unvisible_pnts = ~visible_pnts
        gaussians.prune_points(unvisible_pnts, 0)

def training(dataset, opt, pipe, args):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    viewpoint_stack, pseudo_stack = None, None

    first_iter = 0
    accumulated_training_time = 0.0
    resume_python_rng_state = None
    resume_torch_rng_state = None
    resume_cuda_rng_state = None
    resume_viewpoint_uids = None
    tb_writer = prepare_output_and_logger(dataset)
    

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    ema_loss_for_log = 0.0

    gaussians = GaussianModel(args)
    if args.huggingface:
        scene = HFScene(args, gaussians, shuffle=False)
    else:
        scene = Scene(args, gaussians, shuffle=False)

    gaussians.training_setup(opt)
    torch.cuda.empty_cache()

    if checkpoint:
        checkpoint_data = _load_training_checkpoint(checkpoint)
        first_iter = int(checkpoint_data["iteration"])
        accumulated_training_time = float(checkpoint_data.get("training_time_seconds", 0.0))
        gaussians.restore(checkpoint_data["model_params"], opt)
        resume_python_rng_state = checkpoint_data.get("python_rng_state")
        resume_torch_rng_state = checkpoint_data.get("torch_rng_state")
        resume_cuda_rng_state = checkpoint_data.get("cuda_rng_state")
        resume_viewpoint_uids = checkpoint_data.get("viewpoint_uids")

    if resume_python_rng_state is not None:
        random.setstate(resume_python_rng_state)
    if resume_torch_rng_state is not None:
        torch.set_rng_state(resume_torch_rng_state)
    if resume_cuda_rng_state is not None:
        torch.cuda.set_rng_state(resume_cuda_rng_state)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    ema_loss_for_log = 0.0
    first_iter += 1
    viewpoint_stack = None
    if resume_viewpoint_uids is not None:
        cameras_by_uid = {camera.uid: camera for camera in scene.getTrainCameras()}
        if len(cameras_by_uid) != len(scene.getTrainCameras()):
            raise ValueError("Cannot restore viewpoint stack because training camera uids are not unique")
        try:
            viewpoint_stack = [cameras_by_uid[uid] for uid in resume_viewpoint_uids]
        except KeyError as error:
            raise ValueError(f"Checkpoint contains unknown training camera uid: {error.args[0]}") from error

    training_timer_start = time.perf_counter()
    excluded_training_overhead = 0.0


    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        gt_image = viewpoint_cam.original_image.cuda()

        if args.dataset_type == 'dtu':
            if 'scan110' not in args.source_path :
                bg_mask = (gt_image.max(0, keepdim=True).values < 30/255)
            else:
                bg_mask = (gt_image.max(0, keepdim=True).values < 15/255)

            bg_mask_clone = bg_mask.clone()
            for i in range(1, 50):
                bg_mask[:, i:] *= bg_mask_clone[:, :-i]
            gt_image[bg_mask.repeat(3,1,1)] = 0.

            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            (render_pkg["alpha"][bg_mask]**2).mean().backward()
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
        elif args.dataset_type == 'blender':
            bg_mask = (gt_image.min(0, keepdim=True).values > 254/255)

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        Ll1 =  l1_loss_mask(image, gt_image)
        loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))


        rendered_depth = render_pkg["depth"]
        flow_depth = viewpoint_cam.flow_depth.cuda().unsqueeze(0)


        if rendered_depth.shape[0] != 0 and iteration > 0 and opt.depth_weight > 0:
            midas_depth = viewpoint_cam.midas_depth.cuda().squeeze().unsqueeze(0)
            if args.dataset_type == 'dtu':

                flow_depth[bg_mask] = flow_depth[~bg_mask].mean()
                rendered_depth[bg_mask] = rendered_depth[~bg_mask].mean().detach()
            elif args.dataset_type == 'blender':
                flow_depth[bg_mask] = 0
                midas_depth[bg_mask] = 0

            flow_depth = flow_depth.view(-1,1)
            rendered_depth = rendered_depth.view(-1,1)

            depth_loss = l1_loss_mask(flow_depth, rendered_depth)
            loss += opt.depth_weight * depth_loss
        


        loss.backward(retain_graph=True)
        with torch.no_grad():
            # Progress bar
            if not loss.isnan():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Points": f"{gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            clean_iterations = []
            if args.dataset_type == 'dtu' or args.dataset_type == 'blender':
                clean_iterations = testing_iterations + [first_iter]
                clean_views(iteration, clean_iterations, scene, gaussians, pipe, background)
            # Log and save
            is_full_evaluation = args.full_eval_metrics and iteration in testing_iterations
            training_time_seconds = None
            evaluation_overhead_start = None
            if is_full_evaluation:
                torch.cuda.synchronize()
                evaluation_overhead_start = time.perf_counter()
                training_time_seconds = (
                    accumulated_training_time
                    + evaluation_overhead_start
                    - training_timer_start
                    - excluded_training_overhead
                )
            training_report(args, tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, render, (pipe, background),
                            training_time_seconds)
            if is_full_evaluation:
                torch.cuda.synchronize()
                excluded_training_overhead += time.perf_counter() - evaluation_overhead_start

            if iteration > first_iter and (iteration in saving_iterations):
                save_overhead_start = None
                if args.full_eval_metrics:
                    torch.cuda.synchronize()
                    save_overhead_start = time.perf_counter()
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if args.full_eval_metrics:
                    torch.cuda.synchronize()
                    excluded_training_overhead += time.perf_counter() - save_overhead_start

            # Densification
            if  iteration < opt.densify_until_iter and iteration not in clean_iterations:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = opt.size_threshold
                    
                    if args.dataset_type == 'blender':
                        shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
                        dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
                        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                        sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                        color = torch.clamp_min(sh2rgb + 0.5, 0.0)
                        white_mask = color.min(-1, keepdim=True).values > 253/255
                        gaussians.xyz_gradient_accum[white_mask] = 0
                        gaussians._opacity[white_mask] = gaussians.inverse_opacity_activation(gaussians.opacity_activation(gaussians._opacity[white_mask]) * 0.1)

                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration, opt.dis_prune, opt.split_num)

                    if args.dataset_type == 'blender':
                        if 'ship' in args.source_path: 
                            gaussians.prune_points(gaussians.get_xyz[:,-1] < -0.5, 0)
                        if 'hotdog' in args.source_path: 
                            gaussians.prune_points(gaussians.get_xyz[:,-1] < -0.2, 0)      

                if iteration > opt.densify_from_iter and iteration % opt.prune_interval == 0 and opt.prune_interval != -1:
                    gaussians.prune(args, viewpoint_cam, rendered_depth, opt.prune_depth_threshold)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            gaussians.update_learning_rate(iteration)
            if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                    iteration > args.start_sample_pseudo:
                gaussians.reset_opacity()

            if iteration in checkpoint_iterations:
                checkpoint_overhead_start = None
                checkpoint_training_time = accumulated_training_time
                if args.full_eval_metrics:
                    torch.cuda.synchronize()
                    checkpoint_overhead_start = time.perf_counter()
                    checkpoint_training_time += (
                        checkpoint_overhead_start
                        - training_timer_start
                        - excluded_training_overhead
                    )
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                checkpoint_payload = {
                    "format_version": 2,
                    "model_params": gaussians.capture(),
                    "iteration": iteration,
                    "training_time_seconds": checkpoint_training_time,
                    "python_rng_state": random.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state(),
                    "viewpoint_uids": [camera.uid for camera in viewpoint_stack],
                }
                checkpoint_path = os.path.join(scene.model_path, f"chkpnt{iteration}.pth")
                temporary_checkpoint_path = checkpoint_path + ".tmp"
                torch.save(checkpoint_payload, temporary_checkpoint_path)
                os.replace(temporary_checkpoint_path, checkpoint_path)
                if args.full_eval_metrics:
                    torch.cuda.synchronize()
                    excluded_training_overhead += time.perf_counter() - checkpoint_overhead_start


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(args, tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations,
                    scene, renderFunc, renderArgs, training_time_seconds=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        full_eval_metrics = args.full_eval_metrics and training_time_seconds is not None
        cpu_rng_state = torch.get_rng_state() if full_eval_metrics else None
        cuda_rng_state = torch.cuda.get_rng_state() if full_eval_metrics else None
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                per_view_metrics = []
                is_full_test_eval = full_eval_metrics and config['name'] == 'test'
                staging_dir = None
                render_dir = None
                gt_dir = None
                if is_full_test_eval:
                    output_dir = os.path.join(args.model_path, 'test', f'ours_{iteration}')
                    staging_dir = output_dir + '.tmp'
                    if os.path.isdir(staging_dir):
                        shutil.rmtree(staging_dir)
                    render_dir = os.path.join(staging_dir, 'renders')
                    gt_dir = os.path.join(staging_dir, 'gt')
                    os.makedirs(render_dir, exist_ok=True)
                    os.makedirs(gt_dir, exist_ok=True)
                for idx, viewpoint in enumerate(config['cameras']):
                    render_results = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_results["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    _l1 = l1_loss(image, gt_image).mean().double()
                    l1_test += _l1

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips
                    if is_full_test_eval:
                        per_view_metrics.append(
                            {"psnr": _psnr, "ssim": _ssim, "lpips": _lpips, "l1": _l1}
                        )
                    if is_full_test_eval:
                        save_image(image, os.path.join(render_dir, viewpoint.image_name + '.png'))
                        save_image(gt_image, os.path.join(gt_dir, viewpoint.image_name + '.png'))
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                if is_full_test_eval:
                    if len(per_view_metrics) < NOVEL_VIEW_COUNT:
                        raise RuntimeError(
                            f"test 视图少于 {NOVEL_VIEW_COUNT}，无法统计新视角指标"
                        )
                    all_metrics = {
                        "psnr": psnr_test.item(),
                        "ssim": ssim_test.item(),
                        "lpips": lpips_test.item(),
                        "l1": l1_test.item(),
                    }
                    novel_metrics = {
                        key: (
                            sum(record[key] for record in per_view_metrics[:NOVEL_VIEW_COUNT])
                            / NOVEL_VIEW_COUNT
                        ).item()
                        for key in ("psnr", "ssim", "lpips", "l1")
                    }
                    output_dir = os.path.join(args.model_path, 'test', f'ours_{iteration}')
                    if os.path.isdir(output_dir):
                        shutil.rmtree(output_dir)
                    os.replace(staging_dir, output_dir)
                    evaluation_record = {
                        "format_version": EVALUATION_FORMAT_VERSION,
                        "iteration": iteration,
                        "split": "test",
                        "num_views": len(config['cameras']),
                        "metrics": all_metrics,
                        "view_groups": {
                            ALL_18_VIEW_GROUP: {
                                "num_views": len(config['cameras']),
                                "metrics": dict(all_metrics),
                                "source": "training_in_memory",
                            },
                            NOVEL_12_VIEW_GROUP: {
                                "num_views": NOVEL_VIEW_COUNT,
                                "metrics": novel_metrics,
                                "source": "training_in_memory",
                            },
                        },
                        "training_time_seconds": float(training_time_seconds),
                    }
                    evaluation_path = os.path.join(
                        args.model_path, 'evaluation', f'iteration_{iteration}.json'
                    )
                    _atomic_write_json(evaluation_path, evaluation_record)
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} TRAIN_TIME {:.3f}s".format(
                        iteration, config['name'], l1_test, psnr_test, ssim_test,
                        lpips_test, training_time_seconds))
                    print(
                        "[ITER {}] Evaluating novel_12: L1 {:.7f} PSNR {:.7f} "
                        "SSIM {:.7f} LPIPS {:.7f}".format(
                            iteration,
                            novel_metrics["l1"],
                            novel_metrics["psnr"],
                            novel_metrics["ssim"],
                            novel_metrics["lpips"],
                        )
                    )
                else:
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                        iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                    if is_full_test_eval:
                        tb_writer.add_scalar(
                            config['name'] + '/training_time_seconds', training_time_seconds, iteration
                        )
                        for metric_name, metric_value in novel_metrics.items():
                            tb_writer.add_scalar(
                                NOVEL_12_VIEW_GROUP + '/' + metric_name,
                                metric_value,
                                iteration,
                            )

        if tb_writer:
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        if full_eval_metrics:
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[50_00, 10_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[50_00, 10_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[50_00, 10_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument(
        "--full_eval_metrics",
        action="store_true",
        help="保存完整 test 渲染、结构化指标和不含评估开销的累计训练耗时",
    )
    parser.add_argument("--train_bg", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.test_iterations = sorted(set(args.test_iterations))
    args.save_iterations = sorted(set(args.save_iterations + [args.iterations]))
    args.checkpoint_iterations = sorted(set(args.checkpoint_iterations))

    print(args.test_iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")
