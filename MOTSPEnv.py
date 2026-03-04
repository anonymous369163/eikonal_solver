from dataclasses import dataclass
import torch

# from MOTSProblemDef import get_random_problems, augment_xy_data_by_64_fold_2obj
from MOTSProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)  (row, col) normalized, origin=top-left
    xy_img: torch.Tensor = None
    sat_img: torch.Tensor = None
    distance_matrix: torch.Tensor = None

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class TSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

        self.channels = env_params['in_channels']
        self.img_size = env_params['img_size']
        self.patch_size = env_params['patch_size']
        self.patches = self.img_size // self.patch_size
        self.offsets = torch.tensor([[0, 0]])
        self.distance_matrix = None
        self.sat_images = None

    def load_problems(self, batch_size, aug_factor=1, problems=None, distance_matrix=None, sat_images=None):
        self.batch_size = batch_size
        if problems is not None:
            self.problems = problems
        else:
            self.problems = get_random_problems(batch_size, self.problem_size, num_objectives=1)
        self.distance_matrix = distance_matrix
        self.sat_images = sat_images

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
                if self.distance_matrix is not None:
                    self.distance_matrix = self.distance_matrix.repeat(8, 1, 1)
            else:
                raise NotImplementedError


        # ---------------------------------------------------------------------
        # Coordinate convention alignment:
        # NPZ node coordinates are stored as (x, y) in [0,1] with origin at bottom-left.
        # The image-related parts of this code (xy_img rasterization, PatchEmbedding and SatelliteEncoder
        # positional encodings) use (row, col) with origin at top-left.
        #
        # To keep *all* inputs semantically aligned, we convert once:
        #   (x, y)_bottom_left  ->  (row, col)_top_left  = (1 - y, x)
        # After this, self.problems is always in (row, col) order.
        # ---------------------------------------------------------------------
        coord_dim = self.problems.size(-1)
        if coord_dim % 2 != 0:
            raise ValueError(f"problems last dim must be even (pairs of coords), got {coord_dim}")

        rc_pairs = []
        for k in range(coord_dim // 2):
            xy = self.problems[:, :, 2 * k: 2 * k + 2]
            x = xy[:, :, 0:1]
            y = xy[:, :, 1:2]
            rc = torch.cat((1.0 - y, x), dim=2).clamp(0.0, 1.0)
            rc_pairs.append(rc)
        self.problems = torch.cat(rc_pairs, dim=2)

        device = self.problems.device
        self.offsets = self.offsets.to(device)
        self.xy_img = torch.ones((self.batch_size, self.channels, self.img_size, self.img_size), device=device)
        for i in range(self.channels):
            # self.problems is in (row, col) normalized coordinates after conversion above.
            rc = self.problems[:, :, 2 * i: 2 * i + 2]  # (B, N, 2) -> (row, col)

            # Convert normalized (row, col) to integer pixel indices safely.
            # Use (img_size - 1) so that coord==1.0 maps to the last valid pixel.
            rc_pix = (rc * (self.img_size - 1)).long()
            rc_pix = torch.clamp(rc_pix, 0, self.img_size - 1)

            block_indices = rc_pix // self.patch_size
            self.block_indices = block_indices[:, :, 0] * self.patches + block_indices[:, :, 1]

            rc_pix = rc_pix[:, :, None, :] + self.offsets[None, None, :, :].expand(
                self.batch_size, 1, self.offsets.shape[0], self.offsets.shape[1]
            )
            rc_pix_idx = rc_pix.reshape(-1, 2)
            BATCH_IDX = torch.arange(self.batch_size, device=device)[:, None, None].expand(
                self.batch_size, self.problem_size, self.offsets.shape[0]
            ).reshape(-1)
            self.xy_img[BATCH_IDX, i, rc_pix_idx[:, 0], rc_pix_idx[:, 1]] = 0

        self.BATCH_IDX = torch.arange(self.batch_size, device=device)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size, device=device)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        device = self.problems.device
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long, device=device)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size), device=device)
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        return Reset_State(self.problems, self.xy_img, self.sat_images, self.distance_matrix), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

        # returning values
        done = (self.selected_count == self.problem_size)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        if self.distance_matrix is not None:
            return self._get_travel_distance_from_matrix()
        return self._get_travel_distance_euclidean()

    def _get_travel_distance_from_matrix(self):
        seq = self.selected_node_list
        seq_next = torch.roll(seq, shifts=-1, dims=2)
        batch_idx = torch.arange(self.batch_size, device=seq.device)[:, None, None].expand(-1, seq.size(1), seq.size(2))
        distances = self.distance_matrix[batch_idx, seq, seq_next]
        travel_distances = distances.sum(dim=2)
        return travel_distances.unsqueeze(2)

    def _get_travel_distance_euclidean(self):
        coord_dim = self.problems.size(-1)
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, coord_dim)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, coord_dim)
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq)**2).sum(3).sqrt()
        travel_distances = segment_lengths.sum(2)
        return travel_distances.unsqueeze(2)

