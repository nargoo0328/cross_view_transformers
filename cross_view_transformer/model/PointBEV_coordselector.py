import math
from math import prod
from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn
from cross_view_transformer.util.box_ops import box_cxcywh_to_xyxy

class CoordSelector(nn.Module):
    def __init__(self, spatial_kwargs, voxel_ref, init_buffer=True):
        super().__init__()
        self.spatial_bounds = spatial_kwargs["spatial_bounds"]
        self.spatial_range = spatial_kwargs["projector"]

        assert voxel_ref in ["spatial", "camera"]
        self.voxel_ref = voxel_ref

        self._init_buffer() if init_buffer else None
        return

    def _init_buffer(self):
        self._set_cache_dense_coords()

    def _set_cache_dense_coords(
        self,
    ):
        """Get, and / or, set dense coordinates used during training and validation."""
        # Alias
        X, Y, Z = self.spatial_range
        XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = self.spatial_bounds

        # Coordinates
        if self.voxel_ref == "spatial":
            # (3, rX, rY, Z), r for reverse order.
            dense_vox_coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(XMIN, XMAX, X, dtype=torch.float64),
                    torch.linspace(YMIN, YMAX, Y, dtype=torch.float64),
                    torch.linspace(ZMIN, ZMAX, Z, dtype=torch.float64),
                    indexing="ij",
                )
            ).flip(1, 2)
        self.register_buffer("dense_vox_coords", dense_vox_coords.float())

        # Indices
        dense_vox_idx = torch.stack(
            torch.meshgrid(
                torch.arange(X), torch.arange(Y), torch.arange(Z), indexing="ij"
            )
        ).flip(1, 2)
        self.register_buffer("dense_vox_idx", dense_vox_idx.int())

        return

    def _get_dense_coords(self, bt: int) -> Dict[str, Tensor]:
        """Regular space division of pillars

        Returns:
            vox_coords: 3D voxels coordinates. Voxels can be grouped as regular pillars.
            vox_idx: Corresponding voxels indices.
        """
        vox_coords, vox_idx = self.dense_vox_coords, self.dense_vox_idx
        vox_coords = repeat(vox_coords, "c x y z -> bt c x y z", bt=bt)

        vox_idx = repeat(vox_idx, "c x y z -> bt c x y z", bt=bt)
        return dict(
            {
                "vox_coords": vox_coords,
                "vox_idx": vox_idx,
            }
        )

class SampledCoordSelector(CoordSelector):
    def __init__(self, spatial_kwargs, voxel_ref, coordselec_kwargs={}):
        super().__init__(spatial_kwargs, voxel_ref, init_buffer=False)

        # Init
        self._init_status(coordselec_kwargs)
        self._init_buffer(self.mode, self.val_mode)
        return

    def _init_buffer(self, mode, val_mode):
        self._set_cache_dense_coords()
        X, Y, Z = self.spatial_range
        self._set_cache_grid(mode, val_mode, self.N_coarse, X, Y)

    def _init_status(self, sampled_kwargs):
        # Coarse pass
        self.mode = sampled_kwargs["mode"]
        assert self.mode in [
            # Dense sampling
            "dense",
            # Pillar sampling
            "rnd_pillars",
            "regular_pillars",
            "rnd_patch_pillars",
        ], NotImplementedError("Unsupported mode")
        self.val_mode = sampled_kwargs.get("val_mode", "dense")

        # Coarse
        self.N_coarse = sampled_kwargs["N_coarse"]
        self.patch_size = sampled_kwargs["patch_size"]

        # Fine pass
        self.N_fine = sampled_kwargs["N_fine"]
        self.N_anchor = sampled_kwargs["N_anchor"]
        self.fine_patch_size = sampled_kwargs["fine_patch_size"]
        self.fine_thresh = sampled_kwargs["fine_thresh"]
        return

    # Get voxels.
    def _get_vox_coords_and_idx(
        self, bt, device
    ) -> Dict[str, Tensor]:
        
        # Prepare out
        dict_vox = {
            "vox_coords": None,
            "vox_idx": None,
        }

        # During training
        if self.training:
            if self.mode == "dense":
                dict_out = self._get_dense_coords(bt)
            else:
                dict_out = self._get_sparse_coords(
                    bt,
                    device,
                    self.mode,
                )
        # During validation
        else:
            if self.val_mode == "dense":
                dict_out = self._get_dense_coords(bt)
            # elif (sampling_imgs["lidar"] is not None) and (self.val_mode == "lidar"):
            #     dict_out = self._get_img_pts(sampling_imgs["lidar"], device)
            # elif (sampling_imgs["hdmap"] is not None) and (self.val_mode == "hdmap"):
            #     dict_out = self._get_img_pts(sampling_imgs["hdmap"][:, :, :1], device)
            else:
                dict_out = self._get_sparse_coords(bt, device, self.val_mode)

        dict_vox.update(dict_out)

        return dict_vox

    def _get_img_pts(self, img, device, subsample=True):
        """
        Get point coordinates using an image, for instance a lidar map (projection of lidar points) or an hdmap.
        """
        # Alias
        X, Y, Z = self.spatial_range
        sb = self.spatial_bounds
        assert img.size(0) == 1, "img evaluation only support val_bs=1"

        # From lidar img to xyz
        vox_idx = torch.nonzero(img.squeeze(0).squeeze(0).squeeze(0))[:, -2:]

        # Subsample
        if subsample:
            N_pts = min(self.N_coarse, vox_idx.size(0))
            rnd = torch.randperm(vox_idx.size(0))[:N_pts].to(device)
            vox_idx = torch.index_select(vox_idx, dim=0, index=rnd)

        vox_idx = torch.tensor([X - 1, Y - 1], device=device) - vox_idx
        vox_idx = repeat(vox_idx, "N c -> (N Z) c", Z=Z)
        vox_idx = torch.cat(
            [
                vox_idx,
                repeat(
                    torch.arange(Z, device=device).view(-1, 1),
                    "Z c -> (N Z) c",
                    N=vox_idx.size(0) // Z,
                ),
            ],
            dim=-1,
        )
        scale = torch.tensor(
            [sb[1] - sb[0], sb[3] - sb[2], sb[5] - sb[4]], device=device
        )
        dist = torch.tensor([abs(sb[0]), abs(sb[2]), abs(sb[4])], device=device)
        xyz = torch.tensor([X - 1, Y - 1, Z - 1], device=device)
        vox_coords = (vox_idx / xyz * scale) - dist
        return {
            "vox_coords": repeat(vox_coords, "(xy z) c -> b c xy 1 z", b=1, z=Z),
            "vox_idx": repeat(vox_idx, "(xy z) c -> b c xy 1 z", b=1, z=Z),
        }

    def _set_cache_grid(
        self, mode: str, val_mode: str, N_pts: int, X: int, Y: int
    ) -> None:
        """Get, and / or, set index grid.
        Avoid creating grid at each forward pass.

        Mode:
            - regular_pillars: Regular grid of size (sqrt_Npts,sqrt_Npts) in range [0,1].

            - rnd_pillars, test_*: Regular grid of size (X,Y) in range [0,1].

            - rnd_patch_pillars: Regular grid of size (X,Y) in range [X,Y].
        """
        for m, name in zip([mode, val_mode], ["grid_buffer", "grid_buffer_val"]):
            if m == "regular_pillars":
                sqrtN = int(math.sqrt(N_pts))
                grid = torch.stack(
                    torch.meshgrid(
                        torch.linspace(0, 1, sqrtN),
                        torch.linspace(0, 1, sqrtN),
                        indexing="ij",
                    ),
                    dim=-1,
                )
            elif m in [
                "rnd_pillars",
            ]:
                grid = torch.stack(
                    torch.meshgrid(
                        torch.linspace(0, 1, X), torch.linspace(0, 1, Y), indexing="ij"
                    ),
                    dim=-1,
                )
            elif m in ["rnd_patch_pillars"]:
                grid = torch.stack(
                    torch.meshgrid(
                        torch.arange(0, X), torch.arange(0, Y), indexing="ij"
                    ),
                    dim=-1,
                )
            else:
                grid = None
            self.register_buffer(name, grid)
        return grid

    def _get_sparse_coords(
        self,
        bt: int,
        device: str,
        mode: str = "rnd_pillars",
    ) -> Dict[str, Tensor]:
        """Sample points or pillars in 3D space between 3D spatial bounds.

        Args:
            mode: Select either :
                - "random_pillar" to sample pillars in 3D space.

                - "random_grouped_points" to sample points grouped in window units in 3D space.
                - "random_grouped_pillar" to sample pillars grouped in window units in 3D space.
        """
        # Alias
        X, Y, Z = self.spatial_range
        sb = self.spatial_bounds
        N_coarse = self.N_coarse
        patch_size = self.patch_size
        grid = self.grid_buffer if self.training else self.grid_buffer_val

        # Points:
        if mode == "regular_pillars":
            pillars = repeat(grid, "x y c -> b (x y) c", b=bt)
        elif mode == "rnd_pillars":
            pillars = repeat(grid, "x y c -> b (x y) c", b=bt)
            rnd = torch.randperm(X * Y)[:N_coarse].to(device)
            pillars = torch.index_select(pillars, dim=1, index=rnd)
        elif mode == "rnd_patch_pillars":
            # Get # anchors.
            N_anchors = N_coarse // (patch_size**2)
            perm = torch.randperm(X * Y)[:N_anchors].to(device)
            pillars = repeat(grid, "x y c -> b (x y) c", b=bt)
            pillars_anchor = torch.index_select(pillars, dim=1, index=perm)

            if patch_size != 1:
                flat_idx = pillars_anchor[..., 1] * Y + pillars_anchor[..., 0]
                # Densify
                mask = torch.zeros((bt, X * Y), device=device)
                mask = torch.scatter(mask, 1, flat_idx, 1)
                mask = mask.view(bt, 1, X, Y)
                mask = self._densify_mask(mask, patch_size)
                # Fill remaining points
                xy_vox_idx = self._select_idx_to_keep(mask, N_coarse, (X, Y))
            else:
                xy_vox_idx = pillars_anchor
            div = torch.tensor([X - 1, Y - 1], device=device).view(1, 1, -1)
            pillars = xy_vox_idx / div
        else:
            # Print error
            raise NotImplementedError(f"Unsupported mode: {mode}")

        Nxy = pillars.size(1)
        pillars = repeat(pillars, "bt xy c -> bt c (xy z)", z=Z)

        # -> Regular Z points
        pillar_heights = torch.linspace(0, 1, Z, device=device)
        pillar_heights = repeat(pillar_heights, "z -> bt 1 (xy z)", bt=bt, xy=Nxy)

        # Pillar pts: [0,1]
        pillar_pts = torch.cat([pillars, pillar_heights], dim=1)

        # Voxel coordinates: [-BoundMin, BoundMax]
        scale = torch.tensor(
            [sb[1] - sb[0], sb[3] - sb[2], sb[5] - sb[4]], device=device
        ).view(1, 3, 1)
        dist = torch.tensor([abs(sb[0]), abs(sb[2]), abs(sb[4])], device=device).view(
            1, 3, 1
        )
        vox_coords = pillar_pts * scale - dist

        # Voxel indices: [0,X-1]
        xyz = torch.tensor([X - 1, Y - 1, Z - 1], device=device).view(1, 3, 1)
        vox_idx = (pillar_pts * xyz).round().to(torch.int32)

        # Out
        dict_vox = {
            "vox_coords": rearrange(vox_coords, "b c (xy z) -> b c xy 1 z", z=Z),
            "vox_idx": rearrange(vox_idx, "b c (xy z) -> b c xy 1 z", z=Z),
        }
        return dict_vox

    def _get_flat_idx(self, bt, hw, device):
        flat_idx = torch.stack(
            torch.meshgrid(
                torch.arange(0, bt, device=device),
                torch.arange(0, hw, device=device),
                indexing="ij",
            )
        ).to(dtype=torch.int32)
        return flat_idx

    def _densify_mask(
        self,
        mask: Tensor,
        patch_size: int,
    ) -> Tensor:
        """Augment the mask by convolving it with a kernel of size patch_size. The larger
        the kernel, the more points are considered activated.

        Force: torch.float64 to use nonzero to get indices, otherwise values are nearly zero.
        """
        # Alias
        device = mask.device
        kernel = torch.ones(
            (1, 1, patch_size, patch_size), dtype=torch.float64, device=device
        )
        augm_mask = F.conv2d(
            mask.to(torch.float64), kernel, padding=(patch_size - 1) // 2
        )
        augm_mask = augm_mask.bool()
        augm_mask = rearrange(augm_mask, "bt 1 X Y -> bt (X Y)")
        return augm_mask

    def _select_idx_to_keep(self, mask: Tensor, N_pts: int, X_Y: Tuple[int]) -> Tensor:
        """Select final points to keep.
        Either we keep Nfine points ordered by their importance or we reinject random points when points are
        predicted as not important, otherwise we will have an artefact at the bottom due to the selection
        on uniform null points.
        """
        # Alias
        bt = mask.size(0)
        device = mask.device
        X, Y = X_Y

        out_idx = []
        if N_pts == "dyna":
            for i in range(bt):
                # Numbers of activated elements
                activ_idx = torch.nonzero(mask[i]).squeeze(1)
                out_idx.append(activ_idx)
        else:
            # Reinject random points in batches
            for i in range(bt):
                # Numbers of activated elements
                activ_idx = torch.nonzero(mask[i]).squeeze(1)
                # How many points are not activated.
                n_activ = activ_idx.size(0)
                idle = N_pts - n_activ

                # Less detected points than N_pts
                if idle > 0:
                    # Random selection
                    allowed_idx = torch.nonzero(mask[i] == 0).squeeze(1)
                    perm = torch.randperm(allowed_idx.size(0))
                    augm_idx = allowed_idx[perm[:idle]]
                else:
                    augm_idx = torch.empty([0], device=device, dtype=torch.int64)
                    activ_idx = activ_idx[:N_pts]

                out_idx.append(torch.cat([activ_idx, augm_idx]))

        out_idx = torch.stack(out_idx)
        xy_vox_idx = torch.stack([((out_idx // Y) % X), out_idx % Y], dim=-1)
        return xy_vox_idx

    @torch.no_grad()
    def _get_sampled_fine_coords(self, out, masks) -> Dict[str, Tensor]:
        """Select points according to the coarse pass logit output.
        Args:
            - N_anchor: Number of anchor points to select most relevant locations (highest logits).
            - Patch_size: Size of the patch to select around the anchor points.
        """

        def _parse_output(pred):
            pred_l = []
            for k in pred:
                pred_class = pred[k]
                if not isinstance(pred_class, torch.Tensor):
                    pred_class = pred_class.dense()
                pred_l.append(pred_class)
            out_score = torch.stack(pred_l, dim=1)
            return out_score.sigmoid().max(dim=1)[0]
        
        # Alias
        sb = self.spatial_bounds
        N_anchor: str | int = self.N_anchor
        patch_size = self.fine_patch_size
        X, Y, Z = self.spatial_range
        N_fine = self.N_fine

        # assert 'bev' in out
        # out_score = out['bev'].dense()
        # out_score = (out_score[:, None].sigmoid() * masks[:, None]).flip(-2, -1)
        out_score = _parse_output(out)
        out_score = (out_score[:, None] * masks[:, None]).flip(-2, -1)

        b, t, _, h, w = out_score.shape
        device = out_score.device
        out_score = out_score[:, :, 0]
        # if key == "binimg":
        #     out_score = out_score[:, :, 0]
        # elif key == "hdmap":
        #     out_score = out_score.max(2).values
        out_score = rearrange(out_score, "b t h w -> (b t) (h w)")
        # Indices stop gradients, i.e no gradient backpropagation.
        flat_idx = self._get_flat_idx(b * t, h * w, device)
        flat_idx = rearrange(flat_idx, "c bt hw -> bt hw c", c=2)
        mask = self._get_fine_mask(out_score, flat_idx, (X, Y), N_anchor)
        mask = self._densify_mask(mask, patch_size)

        xy_vox_idx = self._select_idx_to_keep(mask, N_fine, (X, Y))

        xy_vox_idx = repeat(xy_vox_idx, "bt N_fine c -> bt N_fine z c", z=Z, c=2)
        z_vox_idx = torch.arange(Z, device=device)
        z_vox_idx = repeat(
            z_vox_idx, "z -> bt N_fine z 1", N_fine=xy_vox_idx.size(1), bt=b * t
        )
        vox_idx = torch.cat([xy_vox_idx, z_vox_idx], dim=-1).to(dtype=torch.int32)
        vox_idx = rearrange(vox_idx, "bt N_fine z c -> bt c (N_fine z)", c=3)

        # Corresponding points
        vox_coords = vox_idx / torch.tensor([X - 1, Y - 1, Z - 1], device=device).view(
            1, 3, 1
        )
        scale = torch.tensor(
            [sb[1] - sb[0], sb[3] - sb[2], sb[5] - sb[4]], device=device
        ).view(1, 3, 1)
        dist = torch.tensor([abs(sb[0]), abs(sb[2]), abs(sb[4])], device=device).view(
            1, 3, 1
        )
        vox_coords = vox_coords * scale - dist

        # Out
        dict_vox = {
            "vox_coords": rearrange(vox_coords, "b c (xy z) -> b c xy 1 z", z=Z),
            "vox_idx": rearrange(vox_idx, "b c (xy z) -> b c xy 1 z", z=Z),
        }
        return dict_vox
    
    def _get_box_mask(self, pred_boxes, pred_logits, view):
        device = pred_boxes.device
        h, w = self.spatial_range[:2]

        scores, _ = pred_logits.softmax(-1)[..., :-1].max(-1)
        filter_idx = torch.topk(scores, k=60, dim=-1).indices

        # Expand dimensions for filter_idx for matching with pred_boxes_coords
        filter_idx_expand = filter_idx.unsqueeze(-1).expand(*filter_idx.shape, pred_boxes.shape[-1])
        pred_boxes = torch.gather(pred_boxes, 1, filter_idx_expand)

        # project box from lidar to bev
        pred_boxes = pred_boxes[..., :4]
        pred_boxes[..., 2:4] = pred_boxes[..., 2:4].exp()
        pred_boxes_coords = box_cxcywh_to_xyxy(pred_boxes, transform=True)

        pred_boxes_coords = nn.functional.pad(pred_boxes_coords,(0, 1), value=1) # b filter_N 3
        pred_boxes_coords = (torch.einsum('b i j, b N j -> b N i', view, pred_boxes_coords)[..., :4]).int()
        # pad with box
        yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))

        # Expand dimensions for xx and yy to match the dimensions of box
        xx = xx[None, None, ...]
        yy = yy[None, None, ...]

        # Check if the coordinates are inside the boxes
        box_mask = (xx >= pred_boxes_coords[..., 0, None, None]) & (xx <= pred_boxes_coords[..., 2, None, None]) & \
                (yy >= pred_boxes_coords[..., 1, None, None]) & (yy <= pred_boxes_coords[..., 3, None, None])

        # Combine the masks for different boxes using logical OR
        mask = box_mask.any(dim=1)
        return mask

    def _get_fine_mask(
        self,
        out_score,
        flat_idx,
        X_Y: Tuple[int, int],
        N_anchor: Union[str, int],
    ):
        """Initialize mask indexes as either top N_anchor or all points above a threshold.

        Args:
            - N_anchor: can be either a number indicating the number of anchor points or a tag indicating we keep
            points above 0.5 corresponding to the positive class in BCE.
        """
        # Alias
        device = out_score.device
        X, Y = X_Y
        bt, hw, _ = flat_idx.shape

        # Keep all important points.
        if N_anchor == "dyna":
            out_score_flat = torch.nonzero(out_score > self.fine_thresh)
            indices = out_score_flat[:, 0] * X * Y + out_score_flat[:, 1]

            mask = torch.zeros((bt * X * Y), device=device)
            mask = torch.scatter(mask, 0, indices.long(), 1)
            mask = mask.view(bt, X, Y)

        # Keep top N_anchor points.
        else:
            out_idx = out_score.topk(k=N_anchor, dim=1, largest=True).indices
            out_idx = rearrange(out_idx, "bt N -> (bt N)")
            batch_idx = torch.arange(bt, device=device).repeat_interleave(N_anchor)

            # Offset correction
            out_idx = batch_idx * prod(X_Y) + out_idx

            A, B = flat_idx, out_idx
            A = rearrange(A, "bt N c -> (bt N) c", c=2)
            xy_vox_idx = torch.index_select(A, 0, B)
            xy_vox_idx = rearrange(xy_vox_idx, "(bt N) c -> bt N c", c=2, bt=bt)

            # Convolutional kernel
            mask = torch.zeros((bt, X * Y), device=device)
            mask = torch.scatter(mask, 1, xy_vox_idx[..., 1].long(), 1)
            mask = mask.view(bt, X, Y)

        mask = rearrange(mask, "bt X Y -> bt 1 X Y")
        return mask