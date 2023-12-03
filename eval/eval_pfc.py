import argparse
import glob
import os
import sys
import pickle

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
import torch
from vis import SMPLSkeleton, skeleton_render
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle)


def calc_physical_score(dir):
    scores = []
    names = []
    accelerations = []
    up_dir = 2  # z is up
    flat_dirs = [i for i in range(3) if i != up_dir]
    DT = 1 / 30

    it = glob.glob(os.path.join(dir, "*.pkl"))

    smpl = SMPLSkeleton()
    debug_mode = False

    if len(it) > 1000:
        it = random.sample(it, 1000)

    for pkl in tqdm(it):
        info = pickle.load(open(pkl, "rb"))

        if "full_pose" in info.keys():
            joint3d = info["full_pose"]
        else:
            # for GT pkl
            root_pos = torch.Tensor(info["pos"])   # N x 3
            local_q = torch.Tensor(info["q"])      # N x 72

            # to ax
            sq, c = local_q.shape
            local_q = local_q.reshape((1, sq, -1, 3))
            root_pos = root_pos.unsqueeze(0)

            # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
            root_q = local_q[:, :, :1, :]  # sequence x 1 x 3
            root_q_quat = axis_angle_to_quaternion(root_q)
            rotation = torch.Tensor(
                [0.7071068, 0.7071068, 0, 0]
            )  # 90 degrees about the x axis
            root_q_quat = quaternion_multiply(rotation, root_q_quat)
            root_q = quaternion_to_axis_angle(root_q_quat)
            local_q[:, :, :1, :] = root_q

            # don't forget to rotate the root position too 😩
            pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
            root_pos = pos_rotation.transform_points(
                root_pos
            )  # basically (y, z) -> (-z, y), expressed as a rotation for readability
            
            positions = smpl.forward(local_q, root_pos).detach().cpu().numpy()
            joint3d = positions[0]

        # visualiaztion for pose debug
        if debug_mode:
            skeleton_render(
                joint3d,
                out="renders",
                name=pkl,
                sound=False
            )

        root_v = (joint3d[1:, 0, :] - joint3d[:-1, 0, :]) / DT  # root velocity (S-1, 3)
        root_a = (root_v[1:] - root_v[:-1]) / DT  # (S-2, 3) root accelerations
        # clamp the up-direction of root acceleration
        root_a[:, up_dir] = np.maximum(root_a[:, up_dir], 0)  # (S-2, 3)
        # l2 norm
        root_a = np.linalg.norm(root_a, axis=-1)  # (S-2,)
        scaling = root_a.max()
        root_a /= scaling

        foot_idx = [7, 10, 8, 11]
        feet = joint3d[:, foot_idx]  # foot positions (S, 4, 3)
        foot_v = np.linalg.norm(
            feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1
        )  # (S-2, 4) horizontal velocity
        foot_mins = np.zeros((len(foot_v), 2))
        foot_mins[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1])
        foot_mins[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3])

        foot_loss = (
            foot_mins[:, 0] * foot_mins[:, 1] * root_a
        )  # min leftv * min rightv * root_a (S-2,)
        foot_loss = foot_loss.mean()
        scores.append(foot_loss)
        names.append(pkl)
        accelerations.append(foot_mins[:, 0].mean())

    out = np.mean(scores) * 10000
    print(f"{dir} has a mean PFC of {out}")


def parse_eval_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_path",
        type=str,
        default="eval/motions/",
        help="Where to load saved motions",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_eval_opt()
    calc_physical_score(opt.motion_path)
