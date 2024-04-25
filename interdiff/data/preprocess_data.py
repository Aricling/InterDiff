import numpy as np
import os
import json
import sys

sys.path.append("/data1/guoling/InterDiff/interdiff")
from libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer
import torch
import random
from typing import List
from scipy.spatial.transform import Rotation
from tqdm import tqdm


poses = None
betas = None
trans = None

verts = None
jtr = None

smpl = None


def calculate_slices2stride_times(t_length, stride=300) -> List:
    clip_period_list = []
    clip_start = 0
    while 1:
        clip_mid_period = random.randint(stride - 100, stride)
        clip_end = clip_start + stride - 1
        if clip_end + 1 > t_length:
            clip_end = t_length - 1
            clip_start = clip_end - stride + 1
            clip_period_list.append([clip_start, clip_end])
            break

        clip_period_list.append([clip_start, clip_end])
        clip_start += clip_mid_period
    return clip_period_list


# frame数只能在object_fit_all.npz中得到
def process_human_sequence(sequence_path, MODEL_PATH):
    with np.load(
        os.path.join(sequence_path, "object_fit_all.npz"), allow_pickle=True
    ) as f:
        obj_angles, obj_trans, frame_times = f["angles"], f["trans"], f["frame_times"]
    with np.load(
        os.path.join(sequence_path, "smpl_fit_all.npz"), allow_pickle=True
    ) as f:
        global poses, betas, trans
        poses, betas, trans = f["poses"], f["betas"], f["trans"]

    info_file = os.path.join(sequence_path, "info.json")
    info = json.load(open(info_file))
    gender = info["gender"]
    obj_name = info["cat"]
    batch_end = len(frame_times)
    # print("frame_times:", frame_times)

    smpl_male = SMPL_Layer(
        center_idx=0,
        gender="male",
        num_betas=10,
        model_root=str(MODEL_PATH),
        hands=True,
    )
    smpl_female = SMPL_Layer(
        center_idx=0,
        gender="female",
        num_betas=10,
        model_root=str(MODEL_PATH),
        hands=True,
    )
    global smpl
    smpl = {"male": smpl_male, "female": smpl_female}[gender].cuda()

    global verts, jtr
    verts, jtr, _, _ = smpl(
        torch.tensor(poses).cuda(),
        th_betas=torch.tensor(betas).cuda(),
        th_trans=torch.tensor(trans).cuda(),
    )
    # print("verts.shape:", verts.shape, "jtr.shape:", jtr.shape)

    output_dir = os.path.join(sequence_path, "m_jtr_clips")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return batch_end, output_dir


def slice_sequence(clip_period_list, output_dir):
    global jtr
    m_jtr_total = jtr.cpu().numpy()
    file_dir = output_dir
    # print(m_jtr_total.shape)
    global poses, betas, trans
    for clip_list in tqdm(clip_period_list, total=len(clip_period_list)):
        # 目前看来这几个变量应该都是cpu上的numpy
        m_jtr_clip = m_jtr_total[clip_list[0] : clip_list[1] + 1]  # 300帧的clip
        poses_clip = poses[clip_list[0] : clip_list[1] + 1]
        betas_clip = betas[clip_list[0] : clip_list[1] + 1]
        trans_clip = trans[clip_list[0] : clip_list[1] + 1]
        # print('1111', poses_clip.shape, trans_clip.shape)
        final_trans_list = []
        for i, m_jtr in tqdm(enumerate(m_jtr_clip), total=m_jtr_clip.shape[0]):
            pelvis = m_jtr_clip[i, 0].copy()
            if i == 0:
                centroid = pelvis
                global_orient = Rotation.from_rotvec(poses_clip[i, :3]).as_matrix()
                rotation_v = np.eye(3).astype(np.float32)
                cos, sin = global_orient[0, 0] / np.sqrt(
                    global_orient[0, 0] ** 2 + global_orient[2, 0] ** 2
                ), global_orient[2, 0] / np.sqrt(
                    global_orient[0, 0] ** 2 + global_orient[2, 0] ** 2
                )
                rotation_v[[0, 2, 0, 2], [0, 2, 2, 0]] = np.array([cos, cos, -sin, sin])
                rotation = np.linalg.inv(rotation_v).astype(np.float32)

            # print('2', trans_clip[i].shape, centroid.shape)
            new_trans = trans_clip[i] - centroid
            # print(new_trans.shape)
            pelvis = pelvis - centroid
            pelvis_original = pelvis - new_trans
            new_trans = (
                np.dot(new_trans + pelvis_original, rotation.T) - pelvis_original
            )
            # print(new_trans.shape)
            pelvis = np.dot(pelvis, rotation.T)

            # # human vertex in the canonical system
            # global verts
            # human_verts_tran = verts[i].copy()[:, :3] - centroid
            # human_verts_tran = np.dot(human_verts_tran, rotation.T)

            # smpl pose parameter in the canonical system
            r_ori = Rotation.from_rotvec(poses_clip[i, :3])
            r_new = Rotation.from_matrix(rotation) * r_ori
            poses_clip[i, :3] = r_new.as_rotvec()

            final_trans_list.append(new_trans[np.newaxis, :])

        final_poses_array = poses_clip
        final_betas_array = betas_clip
        final_trans_array = np.concatenate(final_trans_list, axis=0)

        global smpl
        verts, jtr, _, _ = smpl(
            torch.tensor(final_poses_array).cuda(),
            th_betas=torch.tensor(final_betas_array).cuda(),
            th_trans=torch.tensor(final_trans_array).cuda(),
        )

        final_jtr_array = jtr[:, :24].cpu().numpy()

        # print(final_poses_array.shape, final_betas_array.shape, final_trans_array.shape, final_jtr_array.shape)

        np.savez(
            os.path.join(file_dir, f"m_jtr_{clip_list[0]}_{clip_list[1]}.npy"),
            poses=final_poses_array,
            betas=final_betas_array,
            trans=final_trans_array,
            jtrs=final_jtr_array,
        )


if __name__ == "__main__":
    sequence_path = "/data1/guoling/InterDiff/interdiff/data/behave/sequences/Date01_Sub01_backpack_back"
    MODEL_PATH = "/data1/guoling/InterDiff/interdiff/body_models/smplh"
    sequence_length, output_dir = process_human_sequence(sequence_path, MODEL_PATH)

    print(sequence_length)

    slice_period_list = calculate_slices2stride_times(
        t_length=sequence_length, stride=300
    )
    print(slice_period_list)

    slice_sequence(slice_period_list, output_dir)

    #########################


# seq_path="/data1/guoling/InterDiff/interdiff/data/behave/sequences/Date02_Sub02_tablesmall_lift"

# # with np.load(os.path.join(seq_path, 'object_fit_all.npz'), allow_pickle=True) as f:
# #     obj_angles, obj_trans, frame_times = f['angles'], f['trans'], f['frame_times']

# #     print(obj_angles, obj_angles.shape)
# #     print(obj_trans, obj_trans.shape)
# #     print(frame_times,len(frame_times), type(frame_times))

# with np.load(os.path.join(seq_path, 'smpl_fit_all.npz'), allow_pickle=True) as f:
#         poses, betas, trans = f['poses'], f['betas'], f['trans']

#         print('1',poses, poses.shape)
#         print('2',betas, betas.shape)
#         print('3',trans, trans.shape)

"""
import sys
# print(sys.path)
sys.path.append(os.getcwd())
from libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer
from psbody.mesh import Mesh

object_path="/data1/guoling/InterDiff/interdiff/data/behave/objects/boxlong/boxlong.obj"

mesh_obj=Mesh()
mesh_obj.load_from_obj(object_path)

obj_verts=mesh_obj.v
obj_faces=mesh_obj.f

print(obj_verts.shape, type(obj_verts))
print(obj_faces.shape, type(obj_faces))
"""
