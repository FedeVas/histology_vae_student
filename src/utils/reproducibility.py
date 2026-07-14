import os
import random

import numpy as np
import torch


def seed_everything(
    seed: int,
    deterministic: bool = True,
) -> None:
    """
    Устанавливает seed для Python, NumPy и PyTorch.

    Parameters
    ----------
    seed:
        Целое число для генераторов случайных чисел.
    deterministic:
        Включить более детерминированное выполнение PyTorch.
    """
    if seed < 0:
        raise ValueError("Seed must be a non-negative integer.")

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Может потребоваться некоторым детерминированным CUDA-операциям.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    else:
        torch.use_deterministic_algorithms(False)

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False


def seed_data_loader_worker(worker_id: int) -> None:
    """
    Устанавливает NumPy и Python seed для каждого DataLoader worker.
    """
    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_torch_generator(seed: int) -> torch.Generator:
    """
    Создаёт отдельный генератор для воспроизводимого shuffle.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator