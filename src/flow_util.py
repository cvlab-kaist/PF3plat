import torch
from torch import nn
import cv2
from einops import einsum
from torch.nn.functional import normalize
from packaging import version
from typing import Any, List, Union, Tuple
import numpy as np
import open3d as o3d
import copy
from pytorch3d.ops import corresponding_points_alignment

import torch
from torch.nn import functional as nn_F
from torch.nn.functional import cosine_similarity

import math
from typing import Optional

def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device)))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device)) * -1)

    theta = torch.acos(cos)

    # theta = torch.min(theta, 2*np.pi - theta)

    return theta.mean()
def split_feature(feature,
                  num_splits=2,
                  channel_last=False,
                  ):
    if channel_last:  # [B, H, W, C]
        b, h, w, c = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = feature.view(b, num_splits, h // num_splits, num_splits, w // num_splits, c
                               ).permute(0, 1, 3, 2, 4, 5).reshape(b_new, h_new, w_new, c)  # [B*K*K, H/K, W/K, C]
    else:  # [B, C, H, W]
        b, c, h, w = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = feature.view(b, c, num_splits, h // num_splits, num_splits, w // num_splits
                               ).permute(0, 2, 4, 1, 3, 5).reshape(b_new, c, h_new, w_new)  # [B*K*K, C, H/K, W/K]

    return feature


def merge_splits(splits,
                 num_splits=2,
                 channel_last=False,
                 ):
    if channel_last:  # [B*K*K, H/K, W/K, C]
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        merge = splits.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            new_b, num_splits * h, num_splits * w, c)  # [B, H, W, C]
    else:  # [B*K*K, C, H/K, W/K]
        b, c, h, w = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, c, h, w)
        merge = splits.permute(0, 3, 1, 4, 2, 5).contiguous().view(
            new_b, c, num_splits * h, num_splits * w)  # [B, C, H, W]

    return merge
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x = tensor_list.tensors  # [B, C, H, W]
        # mask = tensor_list.mask  # [B, H, W], input with padding, valid as 0
        b, c, h, w = x.size()
        mask = torch.ones((b, h, w), device=x.device)  # [B, H, W]
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
def softmax_with_temperature(x, beta, d=1):
    r"""SFNet: Learning Object-aware Semantic Flow (Lee et al.)"""
    M, _ = x.max(dim=d, keepdim=True)
    x = x - M  # subtract maximum value for stability
    exp_x = torch.exp(x / beta)
    exp_x_sum = exp_x.sum(dim=d, keepdim=True)
    return exp_x / exp_x_sum
def points_to_hpoints(points):
    n,_=points.shape
    return np.concatenate([points,np.ones([n,1])],1)
def random_se3():
    T = np.eye(4)
    T[0:3,0:3] = random_rotation_matrix()
    t = np.random.rand(3)-0.5
    T[0:3,-1] = t*1000
    return T
def transform_points(pts,transform):
    h,w=transform.shape
    if h==3 and w==3:
        return pts @ transform.T
    if h==3 and w==4:
        return pts @ transform[:,:3].T + transform[:,3:].T
    elif h==4 and w==4:
        return hpoints_to_points(points_to_hpoints(pts) @ transform.T)
    else: raise NotImplementedError
    
def to_homogeneous(points: Union[torch.Tensor, np.ndarray]):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1]+(1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1]+(1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError

def visualize_points(points, image_size=(256, 256), point_radius=2):
    """
    Visualizes points on a blank image.
    
    Args:
    - points (Tensor): The xy coordinates of points to visualize. Shape: [N, 2].
    - image_size (tuple): The size of the output image (height, width).
    - point_radius (int): The radius of each point to draw.
    
    Returns:
    - numpy.ndarray: An image with the points visualized.
    """
    # Create a blank image
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    
    # Convert points from tensor to numpy if not already
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    # Draw each point on the image
    for (x, y) in points:
        x, y = int(x), int(y)
        if 0 <= x < image_size[1] and 0 <= y < image_size[0]:
            image[max(0, y-point_radius):min(y+point_radius, image_size[0]), max(0, x-point_radius):min(x+point_radius, image_size[1])] = 255  # White points
    
    return image
def from_homogeneous(points: Union[torch.Tensor, np.ndarray]):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + 1e-6)
def batch_project_to_other_img(kpi: torch.Tensor, di: torch.Tensor, 
                               Ki: torch.Tensor, Kj: torch.Tensor, 
                               T_itoj: torch.Tensor, 
                               return_depth=False):
    """
    Project pixels of one image to the other. 
    Args:
        kpi: BxNx2 coordinates in pixels of image i
        di: BxN, corresponding depths of image i
        Ki: intrinsics of image i, Bx3x3
        Kj: intrinsics of image j, Bx3x3
        T_itoj: Transform matrix from coordinate system of i to j, Bx4x4
        return_depth: Bool

    Returns:
        kpi_j: Pixels projection in image j, BxNx2
        di_j: Depth of the projections in image j, BxN
    """
    if len(di.shape) == len(kpi.shape):
        # di must be BxNx1
        di = di.squeeze(-1)
    kpi_3d_i = to_homogeneous(kpi) @ torch.inverse(Ki).transpose(-1, -2)
    kpi_3d_i = kpi_3d_i * di[..., None]  # non-homogeneous coordinates
    kpi_3d_j = from_homogeneous(
            to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    kpi_j = from_homogeneous(kpi_3d_j @ Kj.transpose(-1, -2))
    if return_depth:
        
        di_j = kpi_3d_j[..., -1]
        return kpi_j, di_j
    return kpi_j

def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: torch.Tensor
) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb[None].float()


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: torch.Tensor
) -> torch.Tensor:
    """
    This function generates a 2D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - grid: The grid to generate the embedding from.

    Returns:
    - emb: The generated 2D positional embedding.
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=2)  # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: Union[int, Tuple[int, int]], return_grid=False
) -> torch.Tensor:
    """
    This function initializes a grid and generates a 2D positional embedding using sine and cosine functions.
    It is a wrapper of get_2d_sincos_pos_embed_from_grid.
    Args:
    - embed_dim: The embedding dimension.
    - grid_size: The grid size.
    Returns:
    - pos_embed: The generated 2D positional embedding.
    """
    if isinstance(grid_size, tuple):
        grid_size_h, grid_size_w = grid_size
    else:
        grid_size_h = grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float)
    grid_w = torch.arange(grid_size_w, dtype=torch.float)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if return_grid:
        return (
            pos_embed.reshape(1, grid_size_h, grid_size_w, -1).permute(
                0, 3, 1, 2
            ),
            grid,
        )
    return pos_embed.reshape(1, grid_size_h, grid_size_w, -1).permute(
        0, 3, 1, 2
    )
def softmax_with_temperature(x, beta, d=1):
    r"""SFNet: Learning Object-aware Semantic Flow (Lee et al.)"""
    M, _ = x.max(dim=d, keepdim=True)
    x = x - M  # subtract maximum value for stability
    exp_x = torch.exp(x / beta)
    exp_x_sum = exp_x.sum(dim=d, keepdim=True)
    return exp_x / exp_x_sum

def SE3_inverse(P):
    R_inv = P[..., :3, :3].transpose(-2, -1)
    t_inv = -1 * R_inv @ P[..., :3, 3:4]
    bottom_row = P[..., 3:4, :]
    Rt_inv = torch.cat((R_inv, t_inv), dim=-1)
    P_inv = torch.cat((Rt_inv, bottom_row), dim=-2)
    return P_inv
def make_grid(img):
    B, _, hB, wB = img.shape
    xx = torch.arange(0, wB).view(1, -1).repeat(hB, 1)
    yy = torch.arange(0, hB).view(-1, 1).repeat(1, wB)
    xx = xx.view(1, 1, hB, wB).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, hB, wB).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(img.device)
    return grid

def hpoints_to_points(hpoints):
    return hpoints[:,:-1]/hpoints[:,-1:]


def camera_chaining(Ps, confidence, N):
    """Synchronizes cameras by chaining adjacent views:
        P_{0, 3} = P_{2, 3} @ P_{1, 2} @ P_{0, 1}

    Args:
        Ps (dict): Pairwise view estimates Ps[(i, j)] is transform i -> j
        confidence (dict): confidence for pairwise estimates, not used for chaining.
        N (int): number of views

    Returns:
        FloatTensor: synchronzed pairwise transforms (batch, 4N, 4N)
    """
    for i in range(N - 1):
        j = i + 1
        assert (i, j) in Ps

    # (i,j) are left over from the loop above.
    batch, _, _ = Ps[(i, j)].shape
    device = Ps[(i, j)].device

    L = [torch.eye(4, device=device)[None].expand(batch, 4, 4)]
    for i in range(N - 1):
        j = i + 1
        L.append(Ps[(i, j)].float() @ L[-1].float())

    L = torch.stack(L, 1)

    return L

def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                                        mkpts1, color, text, path=None,
                                        show_keypoints=False, margin=10,
                                        opencv_display=False, opencv_title='',
                                        small_text=[]):
    H0, W0 = image0.shape[:2]
    H1, W1 = image1.shape[:2]
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W, 3), np.uint8)
    
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out

class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions: int = 6,
        omega_0: float = 1.0,
        logspace: bool = True,
        append_input: bool = True,
    ) -> None:
        """
        The harmonic embedding layer supports the classical
        Nerf positional encoding described in
        `NeRF <https://arxiv.org/abs/2003.08934>`_
        and the integrated position encoding in
        `MIP-NeRF <https://arxiv.org/abs/2103.13415>`_.

        During the inference you can provide the extra argument `diag_cov`.

        If `diag_cov is None`, it converts
        rays parametrized with a `ray_bundle` to 3D points by
        extending each ray according to the corresponding length.
        Then it converts each feature
        (i.e. vector along the last dimension) in `x`
        into a series of harmonic features `embedding`,
        where for each i in range(dim) the following are present
        in embedding[...]::

            [
                sin(f_1*x[..., i]),
                sin(f_2*x[..., i]),
                ...
                sin(f_N * x[..., i]),
                cos(f_1*x[..., i]),
                cos(f_2*x[..., i]),
                ...
                cos(f_N * x[..., i]),
                x[..., i],              # only present if append_input is True.
            ]

        where N corresponds to `n_harmonic_functions-1`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.


        If `diag_cov is not None`, it approximates
        conical frustums following a ray bundle as gaussians,
        defined by x, the means of the gaussians and diag_cov,
        the diagonal covariances.
        Then it converts each gaussian
        into a series of harmonic features `embedding`,
        where for each i in range(dim) the following are present
        in embedding[...]::

            [
                sin(f_1*x[..., i]) * exp(0.5 * f_1**2 * diag_cov[..., i,]),
                sin(f_2*x[..., i]) * exp(0.5 * f_2**2 * diag_cov[..., i,]),
                ...
                sin(f_N * x[..., i]) * exp(0.5 * f_N**2 * diag_cov[..., i,]),
                cos(f_1*x[..., i]) * exp(0.5 * f_1**2 * diag_cov[..., i,]),
                cos(f_2*x[..., i]) * exp(0.5 * f_2**2 * diag_cov[..., i,]),,
                ...
                cos(f_N * x[..., i]) * exp(0.5 * f_N**2 * diag_cov[..., i,]),
                x[..., i],              # only present if append_input is True.
            ]

        where N equals `n_harmonic_functions-1`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.

        If `logspace==True`, the frequencies `[f_1, ..., f_N]` are
        powers of 2:
            `f_1, ..., f_N = 2**torch.arange(n_harmonic_functions)`

        If `logspace==False`, frequencies are linearly spaced between
        `1.0` and `2**(n_harmonic_functions-1)`:
            `f_1, ..., f_N = torch.linspace(
                1.0, 2**(n_harmonic_functions-1), n_harmonic_functions
            )`

        Note that `x` is also premultiplied by the base frequency `omega_0`
        before evaluating the harmonic functions.

        Args:
            n_harmonic_functions: int, number of harmonic
                features
            omega_0: float, base frequency
            logspace: bool, Whether to space the frequencies in
                logspace or linear space
            append_input: bool, whether to concat the original
                input to the harmonic embedding. If true the
                output is of the form (embed.sin(), embed.cos(), x)
        """
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions, dtype=torch.float32
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer(
            "_frequencies", frequencies * omega_0, persistent=False
        )
        self.register_buffer(
            "_zero_half_pi",
            torch.tensor([0.0, 0.5 * torch.pi]),
            persistent=False,
        )
        self.append_input = append_input

    def forward(
        self, x: torch.Tensor, diag_cov: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            x: tensor of shape [..., dim]
            diag_cov: An optional tensor of shape `(..., dim)`
                representing the diagonal covariance matrices of our Gaussians, joined with x
                as means of the Gaussians.

        Returns:
            embedding: a harmonic embedding of `x` of shape
            [..., (n_harmonic_functions * 2 + int(append_input)) * num_points_per_ray]
        """
        # [..., dim, n_harmonic_functions]
        embed = x[..., None] * self._frequencies
        # [..., 1, dim, n_harmonic_functions] + [2, 1, 1] => [..., 2, dim, n_harmonic_functions]
        embed = embed[..., None, :, :] + self._zero_half_pi[..., None, None]
        # Use the trig identity cos(x) = sin(x + pi/2)
        # and do one vectorized call to sin([x, x+pi/2]) instead of (sin(x), cos(x)).
        embed = embed.sin()
        if diag_cov is not None:
            x_var = diag_cov[..., None] * torch.pow(self._frequencies, 2)
            exp_var = torch.exp(-0.5 * x_var)
            # [..., 2, dim, n_harmonic_functions]
            embed = embed * exp_var[..., None, :, :]

        embed = embed.reshape(*x.shape[:-1], -1)

        if self.append_input:
            return torch.cat([embed, x], dim=-1)
        return embed

    @staticmethod
    def get_output_dim_static(
        input_dims: int, n_harmonic_functions: int, append_input: bool
    ) -> int:
        """
        Utility to help predict the shape of the output of `forward`.

        Args:
            input_dims: length of the last dimension of the input tensor
            n_harmonic_functions: number of embedding frequencies
            append_input: whether or not to concat the original
                input to the harmonic embedding
        Returns:
            int: the length of the last dimension of the output tensor
        """
        return input_dims * (2 * n_harmonic_functions + int(append_input))

    def get_output_dim(self, input_dims: int = 3) -> int:
        """
        Same as above. The default for input_dims is 3 for 3D applications
        which use harmonic embedding for positional encoding,
        so the input might be xyz.
        """
        return self.get_output_dim_static(
            input_dims, len(self._frequencies), self.append_input
        )
class PoseEmbedding(nn.Module):
    def __init__(self, target_dim, n_harmonic_functions=10, append_input=True):
        super().__init__()

        self._emb_pose = HarmonicEmbedding(
            n_harmonic_functions=n_harmonic_functions, append_input=append_input
        )

        self.out_dim = self._emb_pose.get_output_dim(target_dim)

    def forward(self, pose_encoding):
        e_pose_encoding = self._emb_pose(pose_encoding)
        return e_pose_encoding
def camera_synchronization(
    Ps,
    confidence,
    N,
    squares=10,
    so3_projection=True,
    normalize_confidences=True,
    double=True,
    center_first_camera=True,
):
    """Applies the proposed synchronization algorithm where the pairwise matrix
    is formed and iterative matrix multiplication is applied for synchronization.

    Args:
        Ps (dict): Ps[(i, j)] is pairwise estimate for i -> j
        confidence (dict): conf[(i, j)] is confidence in pairwise estimates
        N (int): number of views
        squares (int, optional): number of matrix multipliactions. Defaults to 10.
        so3_projection (bool, optional): reproject onto SO(3) during optimization
        normalize_confidences (bool, optional): normalize conf colum to 1
        double (bool, optional): run optimization in float64; good for stability
        center_first_camera (bool, optional): return cameras around 0 or N/2 view

    Returns:
        FloatTensor: synchronzed pairwise transforms (batch, 4N, 4N)
    """
    # for 2 views, there's only 1 pairwise estimate ... no sync is possible
    
    if N == 2:
        return camera_chaining(Ps, confidence, N)

    _views_all = []
    for i, j in Ps:
        # sanity checks
        assert (i, j) in confidence
        assert i != j
        assert (j, i) not in Ps
        _views_all.append(i)
        _views_all.append(j)

    for vi in range(N):
        assert vi in _views_all, f"View {vi} is not in any pairwise views"
   
    # (i,j) are left over from the loop above.
    batch, _, _ = Ps[(i, j)].shape
    device = Ps[(i, j)].device

    # form conf_matrix; turn it into a 'stochastic' matrix
    no_entry_conf = torch.zeros(batch, device=device)
    conf = [[no_entry_conf for _ in range(N)] for _ in range(N)]

    for i, j in Ps:
        c = confidence[(i, j)]
        conf[i][j] = c
        conf[j][i] = c
        if normalize_confidences:
            conf[i][i] = conf[i][i] + c / 2
            conf[j][j] = conf[j][j] + c / 2

    if not normalize_confidences:
        for i in range(N):
            conf[i][i] = torch.ones_like(no_entry_conf)
   
    conf = torch.stack([torch.stack(conf_row, dim=1) for conf_row in conf], dim=1)
    if normalize_confidences:
        conf = conf / conf.sum(dim=1, keepdim=True).clamp(min=1e-9)
    
    # === Form L matrix ===
    no_entry_P = torch.zeros(batch, 4, 4, device=device)
    diag_entry_P = torch.eye(4, device=device)[None].expand(batch, 4, 4)
    L = [[no_entry_P for i in range(N)] for j in range(N)]
    
    for i in range(N):
        L[i][i] = conf[:, i, i, None, None] * diag_entry_P

    for i, j in Ps:
        c_ij = conf[:, i, j, None, None]
        c_ji = conf[:, j, i, None, None]
        L[i][j] = c_ij * SE3_inverse(Ps[(i, j)])
        L[j][i] = c_ji * Ps[(i, j)]
   
    L = torch.cat([torch.cat(L_row, dim=2) for L_row in L], dim=1)

    if double:  # turn into double to make it more stable
        L = L.double()

    # Raise L to the power of 2**squares
    for _ in range(squares):
        L = L @ L

    L = L.view(batch, N, 4, N, 4)

    if center_first_camera:
        L = L[:, :, :, 0, :]
    else:
        L = L[:, :, :, N // 2, :]

    mass = L[:, :, 3:, 3:]
    # If mass.min() ==0, either the parameter squares neeeds to be larger, or
    # the set of edges (entries in Ps) does not span the set of cameras.
    if mass.min().item() == 0:
        print("Warning: mass.min() == 0")
        return camera_chaining(Ps, confidence, N)
    "2**squares, or the set of edges, is too small"
    L = L / mass.clamp(min=1e-9)

    if so3_projection:
        R_pre = L[:, :, :3, :3]

        U, _, V = torch.svd(R_pre)
        V_t = V.transpose(-1, -2)
        S = torch.det(U @ V_t)
        S = torch.cat(
            [torch.ones(*S.shape, 1, 2, device=device), S[..., None, None]], -1
        )
        R = (U * S.double()) @ V_t
        L = torch.cat([torch.cat([R, L[:, :, :3, 3:]], 3), L[:, :, 3:]], 2)

    L = L.float()
    
    return L
def inlier_counting_3d(X0, X1, R, t, th=0.05):
    
    delta = (X0[:, None] @ torch.from_numpy(R).to(X0.device).float() + torch.from_numpy(t).to(X0.device).float() - X1[:, None]).norm(2, dim=-1)
    inliers = torch.exp(-delta / th)

    return inliers.mean()

def matches_from_flow(flow, binary_mask=None, scaling=1.0):
    """
    Retrieves the pixel coordinates of 'good' matches in source and target images, based on provided flow field
    (relating the target to the source image) and a binary mask indicating where the flow is 'good'.
    Args:
        flow: tensor of shape B, 2, H, W (will be reshaped if it is not the case). Flow field relating the target
              to the source image, defined in the target image coordinate system.
        binary_mask: bool mask corresponding to valid flow vectors, shape B, H, W
        scaling: scalar or list of scalar (horizontal and then vertical direction):
                 scaling factor to apply to the retrieved pixel coordinates in both images.

    Returns:
        pixel coordinates of 'good' matches in tsource image, Nx2 (numpy array)
        pixel coordinates of 'good' matches in the target image, Nx2 (numpy array)
    """

    B, _, hB, wB = flow.shape
    xx = torch.arange(0, wB).view(1, -1).repeat(hB, 1)
    yy = torch.arange(0, hB).view(-1, 1).repeat(1, wB)
    xx = xx.view(1, 1, hB, wB).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, hB, wB).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if flow.is_cuda:
        grid = grid.cuda()
        if binary_mask is not None:
            binary_mask = binary_mask.cuda()

    mapping = flow + grid
    mapping_x = mapping.permute(0, 2, 3, 1)[:, :, :, 0]
    mapping_y = mapping.permute(0, 2, 3, 1)[:, :, :, 1]
    grid_x = grid.permute(0, 2, 3, 1)[:, :, :, 0]
    grid_y = grid.permute(0, 2, 3, 1)[:, :, :, 1]
    if binary_mask is not None:
        pts2 = torch.cat((grid_x[binary_mask].unsqueeze(1),
                        grid_y[binary_mask].unsqueeze(1)), dim=1)
        pts1 = torch.cat((mapping_x[binary_mask].unsqueeze(1),
                        mapping_y[binary_mask].unsqueeze(1)),
                        dim=1)  # convert to mapping and then take the correspondences
    else:
        pts2 = torch.cat((grid_x.unsqueeze(1), grid_y.unsqueeze(1)), dim=1)
        pts1 = torch.cat((mapping_x.unsqueeze(1), mapping_y.unsqueeze(1)), dim=1)
    return pts1.detach().cpu().numpy()*scaling, pts2.detach().cpu().numpy()*scaling

def matches_from_flow_batch(flow, indices, binary_mask=None, scaling=1.0):
    """
    Retrieves the pixel coordinates of 'good' matches in source and target images, based on provided flow field
    (relating the target to the source image) and a binary mask indicating where the flow is 'good'.
    Args:
        flow: tensor of shape B, 2, H, W (will be reshaped if it is not the case). Flow field relating the target
              to the source image, defined in the target image coordinate system.
        binary_mask: bool mask corresponding to valid flow vectors, shape B, H, W
        scaling: scalar or list of scalar (horizontal and then vertical direction):
                 scaling factor to apply to the retrieved pixel coordinates in both images.

    Returns:
        pixel coordinates of 'good' matches in the source image, Nx2 (numpy array)
        pixel coordinates of 'good' matches in the target image, Nx2 (numpy array)
    """

    B, _, hB, wB = flow.shape
    xx = torch.arange(0, wB).view(1, -1).repeat(hB, 1)
    yy = torch.arange(0, hB).view(-1, 1).repeat(1, wB)
    xx = xx.view(1, 1, hB, wB).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, hB, wB).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if flow.is_cuda:
        grid = grid.cuda()
        if binary_mask is not None:
            binary_mask = binary_mask.cuda()

    mapping = flow + grid
    mapping_x = mapping.permute(0, 2, 3, 1)[:, :, :, 0]
    mapping_y = mapping.permute(0, 2, 3, 1)[:, :, :, 1]
    grid_x = grid.permute(0, 2, 3, 1)[:, :, :, 0]
    grid_y = grid.permute(0, 2, 3, 1)[:, :, :, 1]
    
    for i in range(B):
        if binary_mask is not None:
            pts2 = torch.cat((grid_x[i][binary_mask[i]].unsqueeze(1),
                            grid_y[i][binary_mask[i]].unsqueeze(1)), dim=1)
            pts1 = torch.cat((mapping_x[i][binary_mask[i]].unsqueeze(1),
                            mapping_y[i][binary_mask[i]].unsqueeze(1)),
                            dim=1)
        else:
            pts2 = torch.cat((grid_x[i].unsqueeze(1), grid_y[i].unsqueeze(1)), dim=1)
            pts1 = torch.cat((mapping_x[i].unsqueeze(1), mapping_y[i].unsqueeze(1)), dim=1)

        breakpoint()
  
    return pts1*scaling, pts2*scaling
def soft_argmax(corr, beta=0.02):
    r"""SFNet: Learning Object-aware Semantic Flow (Lee et al.)"""

    b, _, h, w = corr.size()

    corr = softmax_with_temperature(corr, beta=0.02, d=1)
    corr = corr.view(-1, h, w, h, w)  # (target hxw) x (source hxw)

    grid_x = corr.sum(dim=1, keepdim=False)  # marginalize to x-coord.
    x_normal = torch.linspace(-1, 1, w).expand(b, w).to(corr.device)
    x_normal = x_normal.view(b, w, 1, 1)
    grid_x = (grid_x * x_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w

    grid_y = corr.sum(dim=2, keepdim=False)  # marginalize to y-coord.
    y_normal = torch.linspace(-1, 1, h).expand(b, h).to(corr.device)
    y_normal = y_normal.view(b, h, 1, 1)
    grid_y = (grid_y * y_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w
    return grid_x, grid_y


def unnormalise_and_convert_mapping_to_flow(map):
    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:, 0, :, :] = (
        (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0
    )  # unormalise
    mapping[:, 1, :, :] = (
        (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0
    )  # unormalise

    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if mapping.is_cuda:
        grid = grid.cuda()
    flow = mapping - grid
    return flow


def kabsch_algorithm(corr_P, corr_Q, corr_W):
    """Runs the weighted kabsh algorithm ...

    Args:
        corr_P (FloatTensor): pointcloud P (batch, N, 3)
        corr_Q (FloatTensor): pointcloud Q (batch, N, 3)
        corr_W (FloatTensor): correspondence weights (batch, N)

    Returns:
        FloatTensor: (batch, 3, 4) estimated registration
    """
    corr_P = corr_P.double()
    corr_Q = corr_Q.double()
    corr_W = corr_W.double().clamp(min=1e-12)

    corr_W = normalize(corr_W, p=2, dim=-1)
    Rt_out = corresponding_points_alignment(corr_P, corr_Q, corr_W)
    return Rt_out

def nn_gather(points, indices):
    # expand indices to same dimensions as points
    
    indices = indices[:, :, None]
    indices = indices.expand(-1, -1, points.shape[2])
    
    return points.gather(1, indices)

def list_knn_gather(xs, idxs):
    # x[0] NMU
    # idxs NLK
    N, L, K = idxs.shape
    M = xs[0].size(1)
    idxs = (
        idxs.flatten(1, 2)
        .add(torch.arange(N, device=xs[0].device)[:, None] * M)
        .flatten(0, 1)
    )
    return [x.flatten(0, 1)[idxs].view(N, L, K, -1) for x in xs]
def align_cpa_ransac(
    corr_P,
    corr_Q,
    weights,
    schedule=[(3, 128)],
    threshold=0.01,
    return_error_check=True,
):
    """Estimate pairwise alignment from a list of correspondences

    Args:
        corr_P (FloatTensor): Correspondnces P
        corr_Q (_type_): _description_
        weights (_type_): _description_
        schedule (list, optional): _description_. Defaults to [(3, 128)].
        threshold (float, optional): _description_. Defaults to 0.1.
        return_new_weights (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # get useful variables
    assert 1 <= len(schedule) <= 2

    corr_P = corr_P.double()
    corr_Q = corr_Q.double()
    weights = weights.double()
    try:
    
        
    
        
        bs = corr_P.size(0)
        n_hot, n_samples = schedule[0]

        idxs = torch.multinomial( weights[:, None].expand(-1, n_samples, -1).flatten(0, 1), n_hot).unflatten(0, (bs, n_samples))
        P, Q, W = list_knn_gather([corr_P, corr_Q, weights[:, :, None]], idxs)
        T = kabsch_algorithm(P.flatten(0, 1), Q.flatten(0, 1), W.view(-1, n_hot))
        R, t = T.R.unflatten(0, (bs, n_samples)), T.T.view(bs, n_samples, 1, 3)
        delta = (corr_P[:, None] @ R + t - corr_Q[:, None]).norm(2, dim=-1)
        inliers = torch.exp(-delta / threshold)
    
        if len(schedule) == 2:  # grow set of inliers?
            n_hot, n_samples = schedule[1]
            iq = inliers.sum(2)

            # pick inlierdistances corresponding to the best Rt (soft)
            idxs = torch.multinomial(iq, n_samples, replacement=True)
            inliers = inliers[torch.arange(bs)[:, None].expand(-1, n_samples), idxs]

            # resample inliers according to fit
            idxs = torch.multinomial(inliers.flatten(0, 1), n_hot).unflatten(
                0, (bs, n_samples)
            )
            P, Q, W = list_knn_gather([corr_P, corr_Q, weights[:, :, None]], idxs)
            T = kabsch_algorithm(P.flatten(0, 1), Q.flatten(0, 1), W.view(-1, n_hot))
            R, t = T.R.unflatten(0, (bs, n_samples)), T.T.view(bs, n_samples, 1, 3)
            delta = (corr_P[:, None] @ R + t - corr_Q[:, None]).norm(2, dim=-1)
            inliers = torch.exp(-delta / threshold)

        n_inliers = inliers.sum(2)
        best = n_inliers.argmax(dim=1)
        inliers = inliers[torch.arange(bs), best]

        inliers = normalize(inliers, dim=-1).clamp(min=1e-7) * inliers.shape[-1]
        new_weights = weights * inliers
        Rt_PtoQ = kabsch_algorithm(corr_P, corr_Q, new_weights)
        Rt_PtoQ = make_Rt(Rt_PtoQ.R, Rt_PtoQ.T)
        error_check = False
    except:
        Rt_PtoQ = kabsch_algorithm(corr_P, corr_Q, weights)
        Rt_PtoQ = make_Rt(Rt_PtoQ.R, Rt_PtoQ.T)
        error_check = True

    
    if return_error_check:
        return Rt_PtoQ.float(), error_check
    else:
        return Rt_PtoQ.float()

def make_Rt(R, t):
    """
    Encode the transformation X -> X @ R + t where X has shape [n,3]
    """
    Rt = torch.cat([R.transpose(-2, -1), t[..., None]], dim=-1)
    pad = torch.zeros_like(Rt[..., 2:3, :])
    pad[..., -1] = 1.0
    Rt = torch.cat((Rt, pad), dim=-2)
    return Rt

def warp(x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        Args:
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy),1).float()

        if x.is_cuda:
            grid = grid.to(flo.device)
        vgrid = grid + flo
        # makes a mapping out of the flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        output = nn.functional.grid_sample(x, vgrid)
        return output
def warp_grid(x, flo, grid):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        Args:
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        
        vgrid = grid + flo
        # makes a mapping out of the flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        output = nn.functional.grid_sample(x, vgrid)
        return output
    
def convert_flow_to_mapping(flow, output_channel_first=True):
    if not isinstance(flow, np.ndarray):
        # torch tensor
        if len(flow.shape) == 4:
            if flow.shape[1] != 2:
                # size is BxHxWx2
                flow = flow.permute(0, 3, 1, 2)

            B, C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if flow.is_cuda:
                grid = grid.cuda()
            mapping = flow + grid # here also channel first
            if not output_channel_first:
                mapping = mapping.permute(0,2,3,1)
        else:
            if flow.shape[0] != 2:
                # size is HxWx2
                flow = flow.permute(2, 0, 1)

            C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if flow.is_cuda:
                grid = grid.cuda()
            mapping = flow + grid # here also channel first
            if not output_channel_first:
                mapping = mapping.permute(1,2,0).float()
        return mapping.float()
    else:
        # here numpy arrays
        if len(flow.shape) == 4:
            if flow.shape[3] != 2:
                # size is Bx2xHxW
                flow = flow.transpose(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = flow.shape[:3]
            mapping = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                mapping[i, :, :, 0] = flow[i, :, :, 0] + X
                mapping[i, :, :, 1] = flow[i, :, :, 1] + Y
            if output_channel_first:
                mapping = mapping.transpose(0,3,1,2)
        else:
            if flow.shape[0] == 2:
                # size is 2xHxW
                flow = flow.transpose(1,2,0)
            # HxWx2
            h_scale, w_scale = flow.shape[:2]
            mapping = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            mapping[:, :, 0] = flow[:, :, 0] + X
            mapping[:, :, 1] = flow[:, :, 1] + Y
            if output_channel_first:
                mapping = mapping.transpose(2, 0, 1)
        return mapping.astype(np.float32)

    
def get_gt_correspondence_mask(flow):
    """Computes the mask of valid flows (that do not match to a pixel outside of the image). """

    mapping = convert_flow_to_mapping(flow, output_channel_first=True)
    if isinstance(mapping, np.ndarray):
        if len(mapping.shape) == 4:
            # shape is B,C,H,W
            b, _, h, w = mapping.shape
            mask_x = np.logical_and(mapping[:, 0] >= 0, mapping[:, 0] <= w-1)
            mask_y = np.logical_and(mapping[:, 1] >= 0, mapping[:, 1] <= h-1)
            mask = np.logical_and(mask_x, mask_y)
        else:
            _, h, w = mapping.shape
            mask_x = np.logical_and(mapping[0] >= 0, mapping[0] <= w - 1)
            mask_y = np.logical_and(mapping[1] >= 0, mapping[1] <= h - 1)
            mask = np.logical_and(mask_x, mask_y)
        mask = mask.astype(np.bool) if version.parse(torch.__version__) >= version.parse("1.1") else mask.astype(np.uint8)
    else:
        if len(mapping.shape) == 4:
            # shape is B,C,H,W
            b, _, h, w = mapping.shape
            mask = mapping[:, 0].ge(0) & mapping[:, 0].le(w-1) & mapping[:, 1].ge(0) & mapping[:, 1].le(h-1)
        else:
            _, h, w = mapping.shape
            mask = mapping[0].ge(0) & mapping[0].le(w-1) & mapping[1].ge(0) & mapping[1].le(h-1)
        mask = mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask.byte()
    return mask

# Example usage

def drawpoint(img1, pts1, img2, pts2, colors):
    # 
   
    for p1, p2, color in zip(pts1, pts2, colors):
    
        color = tuple(color.tolist())

        img1 = cv2.circle((img1)  , tuple(p1), 5, color, -1)
        
        img2 = cv2.circle((img2  ) , tuple(p2), 5, color, -1)


    return img1, img2

def select_gaussians(gaussians,num_views,select_index):
    '''
    Gaussian Parameters
    - covariances : (batch, num_gaussians, 3, 3)
    - harmonics : (batch, num_gaussians, 3, 1)
    - means : (batch, num_gaussians, 3)
    - opacities : (batch, num_gaussians)
    '''

    num_gaussians_per_view = gaussians.means.shape[1] // num_views
    index1 = select_index[0]
    index2 = select_index[1]
    
    selected_gaussians = gaussians.clone()
    selected_gaussians.covariances = torch.cat((selected_gaussians.covariances[:,num_gaussians_per_view*index1:num_gaussians_per_view*(index1+1),:,:] , selected_gaussians.covariances[:,num_gaussians_per_view*index2:num_gaussians_per_view*(index2+1),:]), dim=1)
    selected_gaussians.harmonics = torch.cat((selected_gaussians.harmonics[:,num_gaussians_per_view*index1:num_gaussians_per_view*(index1+1),:,:] , selected_gaussians.harmonics[:,num_gaussians_per_view*index2:num_gaussians_per_view*(index2+1),:]), dim=1)
    selected_gaussians.means = torch.cat((selected_gaussians.means[:,num_gaussians_per_view*index1:num_gaussians_per_view*(index1+1),:] , selected_gaussians.means[:,num_gaussians_per_view*index2:num_gaussians_per_view*(index2+1),:]), dim=1)
    selected_gaussians.opacities = torch.cat((selected_gaussians.opacities[:,num_gaussians_per_view*index1:num_gaussians_per_view*(index1+1)] , selected_gaussians.opacities[:,num_gaussians_per_view*index2:num_gaussians_per_view*(index2+1)]), dim=1)

    return selected_gaussians

def select_cameras(c2w,num_views,select_index):
    '''
    Camera Parameters
    - c2w : (batch, num_views, 4, 4)
    '''
    for i in range(num_views):
        if i not in select_index:
            camera_index = i
            break
    selected_c2w = c2w[:,camera_index,:,:].unsqueeze(1)

    return selected_c2w