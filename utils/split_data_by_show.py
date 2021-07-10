from os.path import join
import random
import pprint
from collections import defaultdict
from utils.basic_utils import \
    get_show_name_from_vid_name, load_jsonl, save_jsonl, \
    save_json, load_json, flat_list_of_lists

random.seed(2021)
split_names = ["train", "val", "test_public", "test_challenge"]

archive_root = "standalone_eval/data_archive"
archive_split_name2data_path_mapping = {
    s: join(archive_root, f"tvr_zh_en_{s}_archive.jsonl")
    for s in split_names
}
archive_show_name2desc_ids_path = join(archive_root, "show_name2desc_ids.json")

release_root = "data"
release_split_name2data_path_mapping = {
    s: join(release_root, f"tvr_zh_en_{s}_release.jsonl")
    for s in split_names
}


def get_desc_ids_by_show(data_path):
    lines = load_jsonl(data_path)
    show_name2desc_ids = defaultdict(list)
    for e in lines:
        show_name = get_show_name_from_vid_name(e["vid_name"])
        show_name2desc_ids[show_name].append(e["desc_id"])
    return show_name2desc_ids


def main_get_desc_ids_by_show():
    show_name2desc_ids_by_split = {}
    for k, v in archive_split_name2data_path_mapping.items():
        show_name2desc_ids_by_split[k] = get_desc_ids_by_show(v)
    pprint.pprint({k: len(v) for k, v in show_name2desc_ids_by_split.items()})
    save_json(show_name2desc_ids_by_split, archive_show_name2desc_ids_path)


def main_split_file():
    excluded_show_name = "friends"
    archive_show_name2desc_ids = load_json(archive_show_name2desc_ids_path)
    for path_mapping in [archive_split_name2data_path_mapping, release_split_name2data_path_mapping]:
        for split_name, split_path in path_mapping.items():
            desc_id2data = {e["desc_id"]: e for e in load_jsonl(split_path)}
            excluded_desc_ids = archive_show_name2desc_ids[split_name][excluded_show_name]
            other_desc_ids = flat_list_of_lists(
                [v for k, v in archive_show_name2desc_ids[split_name].items() if k != excluded_show_name])
            excluded_data = [desc_id2data[k] for k in excluded_desc_ids]
            save_jsonl(excluded_data, split_path.replace(".jsonl", f"_{excluded_show_name}.jsonl"))
            other_data = [desc_id2data[k] for k in other_desc_ids]
            save_jsonl(other_data, split_path.replace(".jsonl", f"_except_{excluded_show_name}.jsonl"))
            print(f"split_path {split_path}, #lines: total {len(desc_id2data)}, "
                  f"excluded {len(excluded_data)}, others {len(other_data)}")


def sample_training():
    # split_path data/tvr_zh_en_train_release.jsonl, #lines: total 87175, excluded 21235, others 65940
    sample_n = 65940
    train_path = release_split_name2data_path_mapping["train"]
    train_data = load_jsonl(train_path)
    sampled_train_data = random.sample(train_data, k=sample_n)
    save_jsonl(sampled_train_data, train_path.replace(".jsonl", f"_sample{65940}.jsonl"))


if __name__ == '__main__':
    # main_get_desc_ids_by_show()
    # main_split_file()
    sample_training()
