import os
import shutil

from red0orange.file import get_image_files


def rename_data(origin_data_root, save_data_root):
    origin_image_paths = get_image_files(origin_data_root)
    origin_image_names = [os.path.basename(i) for i in origin_image_paths]
    dst_image_names = []
    for image_name in origin_image_names:
        name = image_name.rsplit(".", maxsplit=1)[0]
        suffix = image_name.rsplit(".", maxsplit=1)[1]
        if name[-1] == "r":
            dst_image_names.append(name[:-1] + "_color." + suffix)
        elif name[-1] == "d":
            dst_image_names.append(name[:-1] + "_depth." + suffix)
    dst_image_paths = [os.path.join(save_data_root, i) for i in dst_image_names]
    print_data = []
    for ori_image_path, dst_image_path in zip(origin_image_paths, dst_image_paths):
        shutil.copy(ori_image_path, dst_image_path)
        print_data.append(
            os.path.relpath(
                dst_image_path, os.path.dirname(os.path.dirname(dst_image_path))
            ).rsplit("_", maxsplit=1)[0]
        )
        pass
    for i in print_data:
        print(i)


if __name__ == "__main__":
    rename_data(
        "/home/red0orange/github_projects/CenterSnap/nocs_test_subset/Real/my_origin2",
        "/home/red0orange/github_projects/CenterSnap/nocs_test_subset/Real/my_preprocess2",
    )
    pass
