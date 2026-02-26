import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import tifffile

from model import SAMRoute 

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
_DEFAULT_SAM_CKPT = os.path.normpath(os.path.join(_PROJECT_ROOT, "sam_road_repo", "sam_ckpts", "sam_vit_b_01ec64.pth"))
_SAMROAD_CKPT_DEFAULT = "training_outputs/finetune_demo/checkpoints/last-v24.ckpt"

class InferenceConfig:
    def __init__(self):
        self.SAM_VERSION = 'vit_b'
        self.PATCH_SIZE = 512 
        self.NO_SAM = False
        self.USE_SAM_DECODER = False
        self.ENCODER_LORA = False
        self.FREEZE_ENCODER = True  
        self.FOCAL_LOSS = False
        self.TOPONET_VERSION = 'default'  
        self.SAM_CKPT_PATH = _DEFAULT_SAM_CKPT
        self.ROUTE_COST_MODE = 'add'
        self.ROUTE_ADD_ALPHA = 20.0
        self.ROUTE_ADD_GAMMA = 2.0
        
        # 新增这两行，彻底关闭硬阻挡机制！允许波阵面跨越断点！
        self.ROUTE_ADD_BLOCK_ALPHA = 0.0  
        self.ROUTE_BLOCK_TH = 0.0

        self.ROUTE_EIK_DOWNSAMPLE = 4
        # use_roi=False 时在 512×512 全图求解，对角线路径约 724px
        # 红黑棋盘格扫描收敛更快，但保守起见设为 512 确保全图收敛
        self.ROUTE_EIK_ITERS = 512  
        self.ROUTE_CKPT_CHUNK = 10
        self.ROUTE_ROI_MARGIN = 64
        self.LORA_RANK = 4  # 仅 ENCODER_LORA=True 时生效
        
    def get(self, key, default):
        return getattr(self, key, default)

def run_inference_on_tif(tif_path, ckpt_path=None, save_path=None, encoder_lora=False, lora_rank=4):
    ckpt_path = ckpt_path or _SAMROAD_CKPT_DEFAULT
    save_path = save_path or 'tif_inference_result_fixed.png'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = InferenceConfig()
    if encoder_lora:
        config.ENCODER_LORA = True
        config.LORA_RANK = lora_rank
        config.FREEZE_ENCODER = False
        print(f"LoRA 推理模式: rank={lora_rank}")
    model = SAMRoute(config).to(device)

    # 1. 先加载权重，再 eval()（正确顺序）
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        
        # 剥离可能存在的 'model.' 前缀以确保 100% 匹配
        clean_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('model.', '') if k.startswith('model.') else k
            clean_state_dict[new_k] = v
            
        model.load_state_dict(clean_state_dict, strict=False)
        print(f"✓ 已加载清洗后的 SAM-Road 权重: {os.path.basename(ckpt_path)}")
    else:
        print("⚠️ 警告：未找到 SAM-Road 权重，测试无效！")
        return

    model.eval()

    # 2. 读取并【裁剪】遥感图像 (不要 Resize)
    print(f"正在读取遥感图像: {tif_path}")
    img_array = tifffile.imread(tif_path)
    
    if img_array.ndim == 3 and img_array.shape[2] >= 3:
        img_array = img_array[:, :, :3]
    elif img_array.ndim == 3 and img_array.shape[0] >= 3: 
        img_array = img_array[:3, :, :].transpose(1, 2, 0)
        
    # 处理遥感影像常见的 16-bit 格式，防止直接转 uint8 造成的数据截断
    if img_array.dtype == np.uint16:
        img_array = (img_array / 256.0).astype(np.uint8)
    else:
        img_array = img_array.astype(np.uint8)
    
    # 核心修改：取图像中心 512x512 区域，保留物理分辨率
    h, w = img_array.shape[:2]
    ps = config.PATCH_SIZE
    start_y = max(0, (h - ps) // 2)
    start_x = max(0, (w - ps) // 2)
    
    img_crop = img_array[start_y : start_y + ps, start_x : start_x + ps]
    
    # 如果原图竟然比 512 还小，补黑边
    if img_crop.shape[0] < ps or img_crop.shape[1] < ps:
        pad_img = np.zeros((ps, ps, 3), dtype=np.uint8)
        pad_img[:img_crop.shape[0], :img_crop.shape[1]] = img_crop
        img_crop = pad_img

    rgb_tensor = torch.tensor(img_crop, dtype=torch.float32).unsqueeze(0).to(device)

    # 3. 一次性跑 Encoder+Decoder，保留 road_prob tensor（避免第二次重复推理 Encoder）
    print("生成概率掩码...")
    with torch.no_grad():
        _, mask_scores = model._predict_mask_logits_scores(rgb_tensor)
        road_prob_tensor = mask_scores[..., 1]          # [1, H, W]，保留在 GPU 上备用
        road_prob = road_prob_tensor[0].cpu().numpy()   # numpy 版本用于可视化和点选取

    # 动态寻找路网上概率最高的点作为起点和终点，防止陷入 off-road 惩罚区
    # 注意：np.where 按行优先扫描，[0] 是最靠上的道路像素，[-1] 是最靠下的
    y_coords, x_coords = np.where(road_prob > 0.5)
    if len(y_coords) > 10:
        # 取图像上方区域和下方区域各一个道路点，作为起终点
        quarter = len(y_coords) // 4
        src_yx = torch.tensor([[y_coords[quarter], x_coords[quarter]]], dtype=torch.long).to(device)
        tgt_yx = torch.tensor([[y_coords[-quarter - 1], x_coords[-quarter - 1]]], dtype=torch.long).to(device)
    else:
        # 降级方案
        src_yx = torch.tensor([[64, 64]], dtype=torch.long).to(device)
        tgt_yx = torch.tensor([[448, 448]], dtype=torch.long).to(device)

    print(f"自动选取起点: {src_yx.cpu().numpy().tolist()}, 终点: {tgt_yx.cpu().numpy().tolist()}")

    # 4. 直接用已有的 road_prob_tensor 执行 Eikonal 求解，跳过第二次 Encoder 推理
    print("执行 Eikonal 距离场计算...")
    with torch.no_grad():
        from eikonal import make_source_mask
        from model import _eikonal_soft_sweeping_diff

        B, H, W = road_prob_tensor.shape
        cost = model._road_prob_to_cost(road_prob_tensor).to(dtype=torch.float32)
        src_yx_l = src_yx.long().view(B, 1, 2)
        tgt_yx_l = tgt_yx.long().view(B, 2)
        src_mask = make_source_mask(H, W, src_yx_l)

        T = _eikonal_soft_sweeping_diff(
            cost, src_mask, model.route_eik_cfg, model.route_ckpt_chunk
        )  # [B, H, W]

        yy = tgt_yx_l[:, 0].clamp(0, H - 1)
        xx = tgt_yx_l[:, 1].clamp(0, W - 1)
        dist = T[torch.arange(B, device=device), yy, xx]

        dist_field = T[0].cpu().numpy()

    # 5. 可视化
    print("生成可视化...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(img_crop)
    axes[0].set_title("Cropped Satellite Image (Original Res)")
    axes[0].plot(src_yx[0, 1].cpu(), src_yx[0, 0].cpu(), 'g*', markersize=15, label='Start')
    axes[0].plot(tgt_yx[0, 1].cpu(), tgt_yx[0, 0].cpu(), 'r*', markersize=15, label='End')
    axes[0].legend()
    axes[0].axis('off')
    
    im1 = axes[1].imshow(road_prob, cmap='magma', vmin=0, vmax=1)
    axes[1].set_title("Predicted Probability Mask")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].axis('off')
    
    # 过滤掉 INF 以便正确显示色标
    valid_dist = dist_field[dist_field < 1e5]
    vmax = np.percentile(valid_dist, 95) if len(valid_dist) > 0 else 1000
    im2 = axes[2].imshow(dist_field, cmap='viridis', vmax=vmax)
    axes[2].set_title(f"Eikonal Cost Field\n(Predicted Target Cost: {dist[0].item():.2f})")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"验证完成！结果已保存至: {os.path.abspath(save_path)}")
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SAM-Road inference on TIF image")
    p.add_argument("--ckpt", type=str, default=_SAMROAD_CKPT_DEFAULT, help="SAM-Road checkpoint path")
    p.add_argument("--output", type=str, default="tif_inference_result_fixed.png", help="Output figure path")
    p.add_argument("--tif", type=str, default="Gen_dataset_V2/Gen_dataset/19.940688_110.276704/00_20.021516_110.190699_3000.0/crop_20.021516_110.190699_3000.0_z16.tif", help="Target TIF path")
    p.add_argument("--encoder_lora", action="store_true", help="加载 LoRA 训练的 ckpt 时使用")
    p.add_argument("--lora_rank", type=int, default=4, help="LoRA rank，须与训练时一致")
    args = p.parse_args()
    run_inference_on_tif(args.tif, ckpt_path=args.ckpt, save_path=args.output, encoder_lora=args.encoder_lora, lora_rank=args.lora_rank)