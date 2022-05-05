import functools
from copy import copy, deepcopy
# import multiprocessing
import torch.multiprocessing as multiprocessing
from threading import Thread
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch.multiprocessing import Pool
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


class NetDepth(nn.Module):
    def __init__(self, n_chan1):
        super().__init__()
        self.n_chan1 = n_chan1
        self.conv1 = nn.Conv2d(3, n_chan1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chan1, n_chan1 // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(n_chan1 // 2, n_chan1 // 2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4 * 4 * n_chan1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = F.max_pool2d(torch.tanh(self.conv3(out)), 2)
        out = out.view(-1, 4 * 4 * self.n_chan1 // 2)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def _get_model():
    model_path = os.path.join(data_dir, "p1ch7", "birds_vs_airplanes.pt")
    m = NetDepth(32)
    m.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    return m


def _execute_model(model, data, p_name):
    if model is None:
        start_time = time.time()
        model = _get_model()
        print(f"Load Model Cost Time: {time.time() - start_time}")
    if data is None:
        start_time = time.time()
        data = get_data_loader()
        print(f"Load Data Loader Cost Time: {time.time() - start_time}")

    print(f"{p_name} process dataloader id: {id(data)}, model id: {id(model)}")
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for items, _ in data:
            model(items)

    print(f"{p_name} Process Execute Overed Cost Time: {time.time() - start_time}")


def _execute_model_multiprocess(model, data, workers):
    p_list = []
    for index in range(workers):
        p = multiprocessing.Process(target=_execute_model, args=(model, data, f"p_{index}"))
        p_list.append(p)

    for p in p_list:
        p.start()

    for p in p_list:
        p.join()


def _execute_model_threading(model, data, workers):
    t_list = []
    for index in range(workers):
        t = Thread(target=_execute_model, args=(model, data, f"p_{index}"))
        t_list.append(t)

    for t in t_list:
        t.start()

    for t in t_list:
        t.join()


def get_data_loader():
    data_path = os.path.join(data_dir, "p1ch7")
    cifar10 = datasets.CIFAR10(
        data_path,
        train=True,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4915, 0.4823, 0.4468),
                (0.2470, 0.2435, 0.2616),
            )
        ])
    )
    label_map = {0: 0, 2: 1}

    cifar2 = [
        (img, label_map[label])
        for img, label in cifar10
        if label in label_map
    ]

    dl = DataLoader(cifar2, batch_size=64, shuffle=True)

    return dl


def run_with_scene_0():
    print("run with scene 0")
    print(">" * 50)

    d = get_data_loader()
    m = _get_model()
    print(f"main process dataloader id: {id(d)}, model id: {id(m)}")

    _execute_model_multiprocess(m, d, 1)

    print(">" * 50)

    print()
    print("=" * 50)
    print()

    print("*" * 50)

    _execute_model_multiprocess(m, d, 2)

    print("*" * 50)

    del d
    del m


def run_with_scene_1():
    print("run with scene 1")
    print(">" * 50)

    m = _get_model()

    print(f"main process model id: {id(m)}")

    # _execute_model_multiprocess(m, None, 1)
    _execute_model_threading(m, None, 1)

    print(">" * 50)

    print()
    print("=" * 50)
    print()

    print("*" * 50)

    # _execute_model_multiprocess(m, None, 2)
    _execute_model_threading(m, None, 2)

    print("*" * 50)

    del m


def run_with_scene_2():
    print("run with scene 2")
    print(">" * 50)

    d = get_data_loader()
    print(f"main process dataloader id: {id(d)}")

    _execute_model_multiprocess(None, d, 1)

    print(">" * 50)

    print()
    print("=" * 50)
    print()

    print("*" * 50)

    _execute_model_multiprocess(None, d, 2)

    print("*" * 50)

    del d


def run_with_scene_3():
    print("run with scene 3")
    print(">" * 50)

    _execute_model_multiprocess(None, None, 1)

    print(">" * 50)

    print()
    print("=" * 50)
    print()

    print("*" * 50)

    _execute_model_multiprocess(None, None, 2)

    print("*" * 50)


if __name__ == '__main__':
    # run_with_scene_0()
    # 影响性能
    run_with_scene_1()
    # run_with_scene_2()
    # 影响性能
    # run_with_scene_3()
