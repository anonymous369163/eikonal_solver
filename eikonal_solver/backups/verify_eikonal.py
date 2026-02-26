import math
import heapq
import torch
import matplotlib.pyplot as plt

# 假设你把代码保存为 eikonal.py，并可 import
from eikonal import prob_to_cost, eikonal_soft_sweeping, EikonalConfig

def dijkstra_grid_4n(cost_2d: torch.Tensor, src_yx, goal_yx, h=1.0):
    """4邻接 Dijkstra。边权 0.5*(c(u)+c(v))*h。"""
    nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    return _dijkstra_grid(cost_2d, src_yx, goal_yx, h, nbrs)

def dijkstra_grid_8n(cost_2d: torch.Tensor, src_yx, goal_yx, h=1.0):
    """8邻接 Dijkstra。边权 0.5*(c(u)+c(v))*step_len*h，对角步长 sqrt(2)。"""
    nbrs = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
    ]
    return _dijkstra_grid(cost_2d, src_yx, goal_yx, h, nbrs)

def _dijkstra_grid(cost_2d: torch.Tensor, src_yx, goal_yx, h, nbrs):
    """通用 Dijkstra：nbrs = [(dy, dx, step_len), ...]，边权 w = 0.5*(c(u)+c(v))*step_len*h。"""
    cost = cost_2d.detach().cpu()
    H, W = cost.shape
    sy, sx = src_yx
    gy, gx = goal_yx

    INF = 1e30
    dist = [[INF]*W for _ in range(H)]
    dist[sy][sx] = 0.0
    pq = [(0.0, sy, sx)]

    while pq:
        d, y, x = heapq.heappop(pq)
        if d != dist[y][x]:
            continue
        if (y, x) == (gy, gx):
            return d
        for dy, dx, step_len in nbrs:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W:
                w = 0.5 * (float(cost[y, x]) + float(cost[ny, nx])) * step_len * float(h)
                nd = d + w
                if nd < dist[ny][nx]:
                    dist[ny][nx] = nd
                    heapq.heappush(pq, (nd, ny, nx))
    return dist[gy][gx]

def greedy_backtrace(T_2d: torch.Tensor, src_yx, goal_yx, max_steps=100000):
    """
    用 T 做贪心回溯（每步走向 T 更小的邻居），仅用于可视化。
    """
    T = T_2d.detach()
    H, W = T.shape
    y, x = goal_yx
    sy, sx = src_yx
    path = [(y, x)]
    for _ in range(max_steps):
        if (y, x) == (sy, sx):
            break
        best = (y, x)
        bestv = float(T[y, x])
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W:
                v = float(T[ny, nx])
                if v < bestv:
                    bestv = v
                    best = (ny, nx)
        if best == (y, x):
            # 卡住：可能迭代不够、或平滑导致局部平台
            break
        y, x = best
        path.append((y, x))
    path.reverse()
    return path

def main(device="cuda" if torch.cuda.is_available() else "cpu"):
    H, W = 96, 96
    src = (48, 10)
    goal = (48, 85)

    # 1) 构造概率图：大部分是“路”(0.8)，中间放个“障碍块”(0.01)
    prob = torch.full((H, W), 0.8, device=device)
    prob[35:62, 40:56] = 0.01

    # 2) 转 cost，并令其可导
    cost = prob_to_cost(prob, gamma=3.0, offroad_penalty=80.0, road_block_th=0.05)
    cost = cost.requires_grad_(True)

    # 3) 调你的 eikonal solver
    source_mask = torch.zeros((1, H, W), dtype=torch.bool, device=device)
    source_mask[0, src[0], src[1]] = True

    cfg = EikonalConfig(
        n_iters=100,
        h=1.0,
        tau_min=0.10,
        tau_branch=0.10,
        tau_update=0.01,
        large_val=1e6,
        use_redblack=True,
        monotone=True,
        mode="hard_eval",
    )

    T, curve = eikonal_soft_sweeping(cost, source_mask, cfg, return_convergence=True)
    Te = float(T[goal[0], goal[1]].detach().cpu())

    # 收敛诊断：判断是否"迭代次数少"（看 curve 而非猜）
    print("[Convergence] curve[-10:] =", [f"{x:.2e}" for x in curve[-10:]])
    print("[Convergence] last max_delta =", f"{curve[-1]:.2e}")
    # 若 curve[-1] 仍 1e-2~1e-1：未收敛，加 iters 有帮助
    # 若 curve[-1] < 1e-3 但 T 仍明显偏离 Dijkstra：对照模型不一致 + softmin 偏置，加迭代不会更接近

    # 4) Dijkstra 对照：4邻接 vs 8邻接
    Td4 = dijkstra_grid_4n(cost.detach(), src, goal, h=cfg.h)
    Td8 = dijkstra_grid_8n(cost.detach(), src, goal, h=cfg.h)

    print(f"[Eikonal]     T(goal) = {Te:.4f}")
    print(f"[Dijkstra 4n] T(goal) = {Td4:.4f}  rel_err = {abs(Te-Td4)/max(Td4,1e-6):.2%}")
    print(f"[Dijkstra 8n] T(goal) = {Td8:.4f}  rel_err = {abs(Te-Td8)/max(Td8,1e-6):.2%}")

    # 5) 梯度回传验证：让 loss = T(goal)
    loss = T[goal[0], goal[1]]
    loss.backward()
    g = cost.grad.detach().abs()

    # 6) 可视化
    path = greedy_backtrace(T, src, goal)

    plt.figure()
    plt.title("Cost map")
    plt.imshow(cost.detach().cpu(), cmap="magma")
    plt.scatter([src[1], goal[1]], [src[0], goal[0]], s=30)
    plt.savefig("cost_map.png")

    plt.figure()
    plt.title("Distance field T")
    plt.imshow(T.detach().cpu(), cmap="viridis")
    ys = [p[0] for p in path]; xs = [p[1] for p in path]
    plt.plot(xs, ys, linewidth=2)
    plt.savefig("distance_field_T.png")

    plt.figure()
    plt.title("|d T(goal) / d cost| (grad magnitude)")
    plt.imshow(g.cpu(), cmap="inferno")
    plt.savefig("grad_magnitude.png")

if __name__ == "__main__":
    main()