# import numpy as np
# from pathlib import Path
# import sys

# def check_label_bounds(seg_path, valid_min=0, valid_max=12, ignore_values=None):
#     if ignore_values is None:
#         ignore_values = {255}
#     try:
#         seg = np.load(seg_path)
#     except Exception as e:
#         print(f"无法加载文件: {seg_path}, 错误: {e}")
#         return False

#     # 检查 NaN
#     if np.any(np.isnan(seg)):
#         print(f"文件包含 NaN: {seg_path}")
#         return False

#     unique = np.unique(seg)
#     # 找出超出范围的值
#     invalid = unique[(unique < valid_min) | (unique > valid_max)]
#     # 过滤掉允许的忽略值
#     invalid = invalid[~np.isin(invalid, list(ignore_values))]
    
#     if len(invalid) > 0:
#         print(f"越界文件: {seg_path}")
#         print(f"  标签范围: min={seg.min()}, max={seg.max()}")
#         print(f"  非法值: {invalid}")
#         print(f"  所有唯一值: {unique}")
#         return False
#     return True

# def main(data_root):
#     data_root = Path(data_root)
#     seg_files = list(data_root.rglob("segment.npy"))
#     print(f"找到 {len(seg_files)} 个 segment.npy 文件")
#     bad_files = []
#     for seg_file in seg_files:
#         if not check_label_bounds(seg_file):
#             bad_files.append(seg_file)
#     if bad_files:
#         print(f"\n总共 {len(bad_files)} 个文件存在标签越界问题！")
#     else:
#         print("所有文件的标签均在合法范围内。")

# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         main(sys.argv[1])
#     else:
#         main("/root/autodl-tmp/data/s3dis")

# import numpy as np
# from pathlib import Path

# data_root = Path("/root/autodl-tmp/data/s3dis")
# seg_files = list(data_root.rglob("segment.npy"))

# has_neg1 = False
# has_255 = False
# all_labels = set()

# for seg_file in seg_files:
#     seg = np.load(seg_file)
#     unique = np.unique(seg)
#     all_labels.update(unique.tolist())
#     if -1 in unique:
#         has_neg1 = True
#         print(f"文件包含 -1: {seg_file}")
#     if 255 in unique:
#         has_255 = True
#         print(f"文件包含 255: {seg_file}")

# print(f"\n所有文件中的唯一标签值: {sorted(all_labels)}")
# print(f"是否存在 -1: {has_neg1}")
# print(f"是否存在 255: {has_255}")
# #结果
# # 所有文件中的唯一标签值: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# # 是否存在 -1: False
# # 是否存在 255: False、
import numpy as np
from pathlib import Path

data_root = Path("/root/autodl-tmp/data/s3dis")
coord_files = list(data_root.rglob("coord.npy"))
print(f"找到 {len(coord_files)} 个 coord.npy 文件")

for cf in coord_files:
    coord = np.load(cf)
    if np.any(np.isnan(coord)) or np.any(np.isinf(coord)):
        print(f"坐标含 nan/inf: {cf}")
    # 检查坐标范围是否合理（例如 S3DIS 原始坐标通常在 0~50 米内）
    if np.abs(coord).max() > 1000:
        print(f"坐标范围异常大: {cf}, max={np.abs(coord).max()}")