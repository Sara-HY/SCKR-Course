import torch


def MRR(predict, ground):
    mrr = 0
    extracted = {}

    for idx_, glab in enumerate(ground):
        if glab != 0:
            extracted[idx_] = 1

    _, key = torch.sort(predict, 0, True)
    for i, idx_ in enumerate(key.data.tolist()):
        if idx_ in extracted:
            mrr = 1.0 / (i + 1)
            break
    return mrr


def MAP(predict, ground):
    map = 0
    map_idx = 0
    extracted = {}

    for idx_, glab in enumerate(ground):
        if glab != 0:
            extracted[idx_] = 1

    _, key = torch.sort(predict, 0, True)
    for i, idx_ in enumerate(key.data.tolist()):
        if idx_ in extracted:
            map_idx += 1
            map += map_idx / (i + 1)
    map = map / map_idx if map_idx != 0 else map
    return map
