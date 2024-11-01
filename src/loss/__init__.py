from .loss import Loss
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_pose import Losspose, LossposeCfgWrapper
from .loss_multissim import LossMultiSSIM, LossMultiSSIMCfgWrapper
LOSSES = {
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossposeCfgWrapper: Losspose,
    LossMultiSSIMCfgWrapper: LossMultiSSIM
}

LossCfgWrapper =  LossLpipsCfgWrapper | LossMseCfgWrapper | LossposeCfgWrapper | LossMultiSSIMCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
