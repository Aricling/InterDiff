import numpy as np

# # file_path="/data1/guoling/HOI-Diff/dataset/raw_behave/Date02_Sub02_toolbox_43/smpl_fit_all.npz"
# file_path="/data1/guoling/HOI-Diff/dataset/raw_behave/Date02_Sub02_monitor_move_8/smpl_fit_all.npz"
# aaa_source=np.load(file_path)

# # np.savetxt('array_data.txt', aaa_source['frame_times'], fmt='%s')

# print(aaa_source['frame_times'])



##########################################################
file_path="/data1/guoling/InterDiff/interdiff/data/behave/sequences/Date03_Sub04_boxtiny_part2/smpl_fit_all.npz"

data=np.load(file_path)

# np.savetxt('/data1/guoling/HOI-Diff/array_data.txt', data['frame_times'], fmt='%s')

print(data['frame_times'])