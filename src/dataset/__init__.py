from torch.utils.data import Dataset

from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .dataset_re10k_test import DatasetRE10kTest, DatasetRE10k_TESTCfg
from .dataset_acid_test import DatasetACIDTest, DatasetACID_TESTCfg
from .dataset_dl3dv import Datasetdl3dvCfg, Datasetdl3dv
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "re10k_test": DatasetRE10kTest,
    "dl3dv": Datasetdl3dv,
    "acid_test": DatasetACIDTest,
}


DatasetCfg = DatasetRE10kCfg | DatasetRE10k_TESTCfg | Datasetdl3dvCfg | DatasetACID_TESTCfg


def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
) -> Dataset:
    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.overfit_to_scene is not None,
        cfg.cameras_are_circular,
        step_tracker,
    )
    
    return DATASETS[cfg.name](cfg, stage, view_sampler)