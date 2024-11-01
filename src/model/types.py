from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    def clone(self):
            return Gaussians(
                means=self.means.clone(),
                covariances=self.covariances.clone(),
                harmonics=self.harmonics.clone(),
                opacities=self.opacities.clone()
            )