import numpy as np
import os
import json
import sys
sys.path.append("/data1/guoling/InterDiff/interdiff")
from libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer
import torch


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

'''
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
'''

def calculate_slices2stride_times(stride=300):
    pass

# frame数只能在object_fit_all.npz中得到
def process_human_sequence(sequence_path, MODEL_PATH):
    with np.load(os.path.join(sequence_path, 'object_fit_all.npz'), allow_pickle=True) as f:
        obj_angles, obj_trans, frame_times = f['angles'], f['trans'], f['frame_times']
    with np.load(os.path.join(sequence_path, 'smpl_fit_all.npz'), allow_pickle=True) as f:
        poses, betas, trans = f['poses'], f['betas'], f['trans']

    info_file = os.path.join(sequence_path, 'info.json')
    info = json.load(open(info_file))
    gender = info['gender']
    obj_name = info['cat']
    batch_end = len(frame_times)

    smpl_male = SMPL_Layer(center_idx=0, gender='male', num_betas=10,
                            model_root=str(MODEL_PATH), hands=True)
    smpl_female = SMPL_Layer(center_idx=0, gender='female', num_betas=10,
                        model_root=str(MODEL_PATH), hands=True)
    smpl = {'male': smpl_male, 'female': smpl_female}[gender]

    verts, jtr, _, _ = smpl(torch.tensor(poses), th_betas=torch.tensor(betas), th_trans=torch.tensor(trans))
    print("verts.shape:",verts.shape,"jtr.shape:", jtr.shape)
    m_jtr=jtr[:,:24,:]

    output_dir=os.path.join(sequence_path, 'm_jtr_clips')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savez(os.path.join(output_dir, "clip.npz"), m_jtr=m_jtr)



if __name__=='__main__':
    sequence_path="/data1/guoling/InterDiff/interdiff/data/behave/sequences/Date01_Sub01_backpack_back"
    MODEL_PATH="/data1/guoling/InterDiff/interdiff/body_models/smplh"
    process_human_sequence(sequence_path, MODEL_PATH)
