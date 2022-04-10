# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import numpy as np
# import bpy
from math import radians
from mathutils import Matrix, Vector, Quaternion, Euler
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from lib.models.vibe import VIBE_Demo
from lib.utils.renderer import Renderer
from lib.dataset.inference import Inference
from lib.utils.smooth_pose import smooth_pose
from lib.data_utils.kp_utils import convert_kps
from lib.utils.pose_tracker import run_posetracker

from lib.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_coords_to_orig_img,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)

MIN_NUM_FRAMES = 25

bone_name_from_index = {
    0 : 'Pelvis',
    1 : 'L_Hip',
    2 : 'R_Hip',
    3 : 'Spine1',
    4 : 'L_Knee',
    5 : 'R_Knee',
    6 : 'Spine2',
    7 : 'L_Ankle',
    8: 'R_Ankle',
    9: 'Spine3',
    10: 'L_Foot',
    11: 'R_Foot',
    12: 'Neck',
    13: 'L_Collar',
    14: 'R_Collar',
    15: 'Head',
    16: 'L_Shoulder',
    17: 'R_Shoulder',
    18: 'L_Elbow',
    19: 'R_Elbow',
    20: 'L_Wrist',
    21: 'R_Wrist',
    22: 'L_Hand',
    23: 'R_Hand'
}

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    video_file = args.vid_file

    # ========= [Optional] download the youtube video ========= #
    # if video_file.startswith('https://www.youtube.com'):
    #     print(f'Donwloading YouTube video \"{video_file}\"')
    #     video_file = download_youtube_clip(video_file, '/tmp')
    #
    #     if video_file is None:
    #         exit('Youtube url is not valid!')
    #
    #     print(f'YouTube Video has been downloaded to {video_file}...')
    #
    # if not os.path.isfile(video_file):
    #     exit(f'Input video \"{video_file}\" does not exist!')

    # ======== read file from rtmp ======== #
    # if not video_file.startswith('rtmp'):
    #     exit('url is not valid')

    # output_path = os.path.join(args.output_folder, os.path.basename(video_file).replace('rtmp://', ''))
    # os.makedirs(output_path, exist_ok=True)

    mot = MPT(
        device=device,
        batch_size=args.tracker_batch_size,
        display=args.display,
        detector_type=args.detector,
        output_format='dict',
        yolo_img_size=args.yolo_img_size,
    )

    # ========= Define VIBE model ========= #
    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)

    # ========= Load pretrained weights ========= #
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')
    # image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)

    # print(f'Input video number of frames {num_frames}')
    # orig_height, orig_width = img_shape[:2]

    total_time = time.time()

    # ========= Read image from rtmp ======== #
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'Input video fps is {fps}')

    orig_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    orig_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    ret, image = cap.read()

    bbox_scale = 1.1
    while ret:

        start = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        tracking_results = mot(image)

        # if tracking_results.shape[0] == 0:
        #     continue

        vibe_results = {}

        for person_id in list(tracking_results.keys()):
            bboxes = joints2d = None

            if args.tracking_method == 'bbox':
                bboxes = tracking_results[person_id]['bbox']
            elif args.tracking_method == 'pose':
                joints2d = tracking_results[person_id]['joints2d']

            frames = tracking_results[person_id]['frames']

            dataset = Inference(
                image=image,
                frames=frames,
                bboxes=bboxes,
                joints2d=joints2d,
                scale=bbox_scale,
            )

            bboxes = dataset.bboxes
            frames = dataset.frames
            has_keypoints = True if joints2d is not None else False

            dataloader = DataLoader(dataset, batch_size=args.vibe_batch_size, num_workers=1)
            with torch.no_grad():
                pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, smpl_joints2d, norm_joints2d = [], [], [], [], [], [], []

                for batch in dataloader:
                    if has_keypoints:
                        batch, nj2d = batch
                        norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                    batch = batch.unsqueeze(0)
                    batch = batch.to(device)

                    batch_size, seqlen = batch.shape[:2]
                    output = model(batch)[-1]

                    pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                    pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                    pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1))
                    pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1))
                    pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))
                    smpl_joints2d.append(output['kp_2d'].reshape(batch_size * seqlen, -1, 2))

                pred_cam = torch.cat(pred_cam, dim=0)
                pred_verts = torch.cat(pred_verts, dim=0)
                pred_pose = torch.cat(pred_pose, dim=0)
                pred_betas = torch.cat(pred_betas, dim=0)
                pred_joints3d = torch.cat(pred_joints3d, dim=0)
                smpl_joints2d = torch.cat(smpl_joints2d, dim=0)
                del batch

            # ========= Save results to a pickle file ========= #
            pred_cam = pred_cam.cpu().numpy()
            pred_verts = pred_verts.cpu().numpy()
            pred_pose = pred_pose.cpu().numpy()
            pred_betas = pred_betas.cpu().numpy()
            pred_joints3d = pred_joints3d.cpu().numpy()
            smpl_joints2d = smpl_joints2d.cpu().numpy()

            # Runs 1 Euro Filter to smooth out the results
            if args.smooth:
                min_cutoff = args.smooth_min_cutoff # 0.004
                beta = args.smooth_beta # 1.5
                print(f'Running smoothing on person {person_id}, min_cutoff: {min_cutoff}, beta: {beta}')
                pred_verts, pred_pose, pred_joints3d = smooth_pose(pred_pose, pred_betas,
                                                                   min_cutoff=min_cutoff, beta=beta)

            orig_cam = convert_crop_cam_to_orig_img(
                cam=pred_cam,
                bbox=bboxes,
                img_width=orig_width,
                img_height=orig_height
            )

            joints2d_img_coord = convert_crop_coords_to_orig_img(
                bbox=bboxes,
                keypoints=smpl_joints2d,
                crop_size=224,
            )

            output_dict = {
                'pred_cam': pred_cam,
                'orig_cam': orig_cam,
                'verts': pred_verts,
                'pose': pred_pose,
                'betas': pred_betas,
                'joints3d': pred_joints3d,
                'joints2d': joints2d,
                'joints2d_img_coord': joints2d_img_coord,
                'bboxes': bboxes,
                'frame_ids': frames,
            }

            vibe_results[person_id] = output_dict

        for person_id in list(vibe_results.keys()):
            quaternion_pose = trans_pose(vibe_results[person_id]['pose'])
            print(quaternion_pose)
        end = time.time()
        print(f'use Time {end - start}')
        ret, image = cap.read()


    # end = time.time()
    # fps = num_frames / (end - vibe_time)

    print(f'VIBE FPS: {fps:.2f}')
    total_time = time.time() - total_time
    print(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
    # print(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

    # print(f'Saving output results to \"{os.path.join(output_path, "vibe_output.pkl")}\".')

    # joblib.dump(vibe_results, os.path.join(output_path, "vibe_output.pkl"))

    print('================= END =================')


def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

def trans_pose(pose):
    # trans = np.zeros((poses.shape[0], 3))

    # bpy.ops.object.mode_set(mode='EDIT')
    # pelvis_position = Vector(bpy.data.armatures[0].edit_bones[bone_name_from_index[0]].head)
    # bpy.ops.object.mode_set(mode='OBJECT')
    pelvis_position = Vector([0, 0, 0])
    # print(pose.shape)
    if pose.shape[1] == 72:
        rod_rots = pose.reshape(24, 3)
    else:
        rod_rots = pose.reshape(26, 3)

    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]

    ret = [0 for _ in range(24)]

    for index, mat_rot in enumerate(mat_rots, 0):
        if index >= 24:
            continue

        bone_rotation = Matrix(mat_rot).to_quaternion()
        quat_x_90_cw = Quaternion((1.0, 0.0, 0.0), radians(-90))
        quat_z_90_cw = Quaternion((0.0, 0.0, 1.0), radians(-90))

        if index == 0:
            # Rotate pelvis so that avatar stands upright and looks along negative Y avis
            ret[index] = (quat_x_90_cw @ quat_z_90_cw) @ bone_rotation
        else:
            ret[index] = bone_rotation

    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_file', type=str,
                        help='input video path or youtube link')

    # parser.add_argument('--output_folder', type=str,
    #                     help='output folder to write results')

    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='batch size of object detector used for bbox tracking')

    parser.add_argument('--staf_dir', type=str, default='/home/mkocabas/developments/openposetrack',
                        help='path to directory STAF pose tracking method installed.')

    parser.add_argument('--vibe_batch_size', type=int, default=20,
                        help='batch size of VIBE')

    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')

    parser.add_argument('--no_render', action='store_true',
                        help='disable final rendering of output video.')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')

    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    parser.add_argument('--smooth', action='store_true',
                        help='smooth the results to prevent jitter')

    parser.add_argument('--smooth_min_cutoff', type=float, default=0.004,
                        help='one euro filter min cutoff. '
                             'Decreasing the minimum cutoff frequency decreases slow speed jitter')

    parser.add_argument('--smooth_beta', type=float, default=0.7,
                        help='one euro filter beta. '
                             'Increasing the speed coefficient(beta) decreases speed lag.')

    args = parser.parse_args()

    main(args)
