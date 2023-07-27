import os
import torch
import argparse
import collections


def load_model(checkpoint_path):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    saved_state_dict = checkpoint_dict["model_g"]
    return saved_state_dict


def save_model(state_dict, checkpoint_path):
    torch.save({'model_g': state_dict}, checkpoint_path)


def average_model(model_list):
    model_keys = list(model_list[0].keys())
    model_average = collections.OrderedDict()
    for key in model_keys:
        key_sum = 0
        for i in range(len(model_list)):
            key_sum = (key_sum + model_list[i][key])
        model_average[key] = torch.div(key_sum, float(len(model_list)))
    return model_average
#   ss_list = []
#   ss_list.append(s1)
#   ss_list.append(s2)
#   ss_merge = average_model(ss_list)


def merge_model(model1, model2, rate):
    model_keys = model1.keys()
    model_merge = collections.OrderedDict()
    for key in model_keys:
        key_merge = rate * model1[key] + (1 - rate) * model2[key]
        model_merge[key] = key_merge
    return model_merge


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m1', '--model1', type=str, required=True)
    parser.add_argument('-m2', '--model2', type=str, required=True)
    parser.add_argument('-r1', '--rate', type=float, required=True)
    args = parser.parse_args()

    print(args.model1)
    print(args.model2)
    print(args.rate)

    assert args.rate > 0 and args.rate < 1, f"{args.rate} should be in range (0, 1)"
    s1 = load_model(args.model1)
    s2 = load_model(args.model2)

    merge = merge_model(s1, s2, args.rate)
    save_model(merge, "sovits5.0_merge.pth")
