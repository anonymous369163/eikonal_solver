"""
Finetune SAMRoute map_decoder on Gen_dataset_V2/Gen_dataset (multi-image batch training).

Optimized for multi-GPU (DDP), large batch, TF32, cached features, high-throughput data loading.
Supports Single-Region Overfitting for debugging.
"""
import argparse
import os
import glob
import torch
import numpy as np
import tifffile
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# 开启 TF32：4090 等 Ada 架构上矩阵乘法可加速 2~3 倍，几乎无精度损失
torch.set_float32_matmul_precision("high")

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger

from model import SAMRoute
from dataset import build_dataloaders

# ==========================================
# 1. 训练配置
# ==========================================
class TrainConfig:
    def __init__(self):
        self.SAM_VERSION = 'vit_b'
        self.PATCH_SIZE = 512
        self.NO_SAM = False
        self.USE_SAM_DECODER = False
        self.ENCODER_LORA = False
        self.LORA_RANK = 4   # 仅当 ENCODER_LORA=True 时生效
        self.FREEZE_ENCODER = True   # 绝对冻结 SAM Encoder，只训练 Decoder（LoRA 模式必须 False）
        self.FOCAL_LOSS = False
        self.TOPONET_VERSION = 'default'

        _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        _PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
        self.SAM_CKPT_PATH = os.path.join(_PROJECT_ROOT, "sam_road_repo", "sam_ckpts", "sam_vit_b_01ec64.pth")

        # 路由/损失配置
        self.ROUTE_COST_MODE = 'add'
        self.ROAD_POS_WEIGHT = 5.0
        
        # 🚀 核心改动：开启双目标损失，加大 Dice 权重，逼迫模型预测极细贴边的道路
        self.ROAD_DICE_WEIGHT = 0.8   # 提高以加强细路约束（原 0.4，细路漏画时尝试 0.6~0.7）
        self.ROAD_DUAL_TARGET = True
        self.ROAD_THIN_BOOST = 10.0    # BCE 细路像素额外权重倍数（原 3.0）

        self.ROUTE_LAMBDA_SEG = 1.0
        self.ROUTE_LAMBDA_DIST = 0.0
        self.BASE_LR = 5e-4  # 配合 batch_size=16，较 4 时适当放大
        self.ENCODER_LR_FACTOR = 0.1   # configure_optimizers 需此属性
        self.LR_MILESTONES = [150] 

    def get(self, key, default):
        return getattr(self, key, default)

# ==========================================
# 2. 单图过拟合专属 Dataset (极速版)
# ==========================================
class SingleRegionDataset(Dataset):
    def __init__(self, region_dir, config, samples_per_epoch=200):
        self.patch_size = config.PATCH_SIZE
        self.steps = samples_per_epoch
        self.config = config
        
        print(f"📦 正在加载单区域数据: {region_dir}")
        # 1. 加载 TIF
        tifs = glob.glob(os.path.join(region_dir, "crop_*.tif"))
        img_array = tifffile.imread(tifs[0])
        if img_array.ndim == 3 and img_array.shape[2] >= 3:
            img_array = img_array[:, :, :3]
        elif img_array.ndim == 3 and img_array.shape[0] >= 3: 
            img_array = img_array[:3, :, :].transpose(1, 2, 0)
        if img_array.dtype == np.uint16:
            img_array = (img_array / 256.0).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
        self.img_array = img_array

        # 2. 加载 Masks (支持双目标)
        masks_thick = glob.glob(os.path.join(region_dir, "roadnet_normalized_*.png"))
        masks_thin = glob.glob(os.path.join(region_dir, "roadnet_*.png"))
        masks_thin = [m for m in masks_thin if "normalized" not in m]
        
        # BCE 寻找宽掩码 (厚度保证高召回)
        mask_path = masks_thick[0] if masks_thick else masks_thin[0]
        self.mask_array = np.array(Image.open(mask_path).convert('L'))
        
        # Dice 寻找细掩码 (细度保证高定位)
        if config.ROAD_DUAL_TARGET and masks_thin:
            self.thin_mask_array = np.array(Image.open(masks_thin[0]).convert('L'))
        else:
            self.thin_mask_array = None

        # 3. 加载缓存特征（LoRA 模式下必须在线计算，不可用缓存）
        feats = glob.glob(os.path.join(region_dir, "samroad_feat_full_*.npy"))
        if getattr(config, 'ENCODER_LORA', False):
            self.feat_array = None
            print("LoRA 模式：跳过 cached_feature，在线计算 Encoder 输出")
        elif feats:
            self.feat_array = np.load(feats[0]).astype(np.float32)
            print("✓ 已找到 cached_feature，开启免 Encoder 极速微调")
        else:
            self.feat_array = None
            print("⚠️ 未找到 cached_feature，将在线跑 Encoder")

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        h, w = self.img_array.shape[:2]
        ps = self.patch_size
        
        y = np.random.randint(0, max(1, h - ps))
        x = np.random.randint(0, max(1, w - ps))

        # 🚀 核心修复：强制对齐到 16 的整数倍，彻底消灭特征图与 Mask 的空间错位！
        y = (y // 16) * 16
        x = (x // 16) * 16
            
        img_crop = self.img_array[y : y + ps, x : x + ps]
        mask_crop = self.mask_array[y : y + ps, x : x + ps]
        
        # 尺寸补齐
        pad_y = max(0, ps - img_crop.shape[0])
        pad_x = max(0, ps - img_crop.shape[1])
        if pad_y > 0 or pad_x > 0:
            img_crop = np.pad(img_crop, ((0, pad_y), (0, pad_x), (0, 0)))
            mask_crop = np.pad(mask_crop, ((0, pad_y), (0, pad_x)))

        sample = {
            'rgb': torch.tensor(img_crop, dtype=torch.float32),
            'road_mask': (torch.tensor(mask_crop, dtype=torch.float32) > 0).float()
        }

        if self.thin_mask_array is not None:
            thin_crop = self.thin_mask_array[y : y + ps, x : x + ps]
            if pad_y > 0 or pad_x > 0:
                thin_crop = np.pad(thin_crop, ((0, pad_y), (0, pad_x)))
            sample['road_mask_thin'] = (torch.tensor(thin_crop, dtype=torch.float32) > 0).float()

        if self.feat_array is not None:
            stride = 16
            feat_size = ps // stride
            fy, fx = y // stride, x // stride
            feat_crop = self.feat_array[:, fy : fy + feat_size, fx : fx + feat_size]
            pad_fy = max(0, feat_size - feat_crop.shape[1])
            pad_fx = max(0, feat_size - feat_crop.shape[2])
            if pad_fy > 0 or pad_fx > 0:
                feat_crop = np.pad(feat_crop, ((0, 0), (0, pad_fy), (0, pad_fx)))
            sample['encoder_feat'] = torch.tensor(feat_crop)

        return sample

# ==========================================
# 3. 训练流程 (使用 Lightning Trainer)
# ==========================================
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"准备训练... 检测到设备: {device}")

    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

    data_root = args.data_root if os.path.isabs(args.data_root) else os.path.join(_PROJECT_ROOT, args.data_root)
    PRETRAINED_CKPT = args.pretrained_ckpt if os.path.isabs(args.pretrained_ckpt) else os.path.join(_PROJECT_ROOT, args.pretrained_ckpt)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(_PROJECT_ROOT, args.output_dir)
    ckpt_dir = os.path.join(output_dir, "checkpoints")

    os.makedirs(ckpt_dir, exist_ok=True)

    config = TrainConfig()
    if args.lr is not None:
        config.BASE_LR = args.lr
    if args.encoder_lora:
        config.ENCODER_LORA = True
        config.LORA_RANK = args.lora_rank
        config.FREEZE_ENCODER = False  # 避免外部冻结误伤 LoRA 参数
        print(f"LoRA 模式: rank={config.LORA_RANK}")
    if args.road_pos_weight is not None:
        config.ROAD_POS_WEIGHT = args.road_pos_weight
    if args.road_dice_weight is not None:
        config.ROAD_DICE_WEIGHT = args.road_dice_weight
    if args.road_thin_boost is not None:
        config.ROAD_THIN_BOOST = args.road_thin_boost
    model = SAMRoute(config)

    # 手动冻结本阶段不参与 Loss 的参数，满足 DDP 校验，从而使用标准 ddp 而非龟速 find_unused_parameters
    if config.FREEZE_ENCODER:
        for p in model.image_encoder.parameters():
            p.requires_grad = False
    if config.ROUTE_LAMBDA_DIST == 0.0:
        model.cost_log_alpha.requires_grad = False
        model.cost_log_gamma.requires_grad = False
        model.eik_gate_logit.requires_grad = False
        if hasattr(model, 'topo_net'):
            for p in model.topo_net.parameters():
                p.requires_grad = False

    # 1. 安全加载预训练权重（若模型结构变化导致无法加载，则从随机初始化继续训练）
    if os.path.isfile(PRETRAINED_CKPT):
        try:
            print(f"加载预训练权重: {PRETRAINED_CKPT}")
            ckpt = torch.load(PRETRAINED_CKPT, map_location='cpu', weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            clean_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict, strict=False)
            print("✓ 预训练权重加载成功")
        except Exception as e:
            print(f"⚠️ 预训练权重加载失败（可能与 map_decoder 结构变更不兼容）: {e}")
            print("   将从随机初始化 Decoder 开始训练")
    else:
        print(f"⚠️ 未找到预训练权重 {PRETRAINED_CKPT}，从随机 Decoder 开始训练。")

    # 2. 构建 DataLoader (自动路由：单图测试 或 全量训练)
    if args.single_region_dir:
        print(f"\n=============================================")
        print(f"⚠️ 注意：当前处于【单图过拟合测试模式】")
        print(f"=============================================\n")
        train_ds = SingleRegionDataset(args.single_region_dir, config, samples_per_epoch=200)
        val_ds = SingleRegionDataset(args.single_region_dir, config, samples_per_epoch=20)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"数据集根目录不存在: {data_root}")
        print("正在构建全量数据集并加载到内存...")
        train_loader, val_loader = build_dataloaders(
            root_dir=data_root,
            patch_size=config.PATCH_SIZE,
            batch_size=args.batch_size,
            num_workers=args.workers,
            include_dist=False,
            val_fraction=args.val_fraction,
            samples_per_region=args.samples_per_region,
            use_cached_features=args.use_cached_features,
            preload_to_ram=args.preload_to_ram,
            road_dilation_radius=args.road_dilation_radius,
        )

    # 3. 配置 Lightning 回调
    ckpt_filename = f'best_{args.run_name}' if args.run_name else 'best'
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=ckpt_filename,
        save_top_k=1,
        monitor='val_seg_loss',
        mode='min',
        save_last=True,
        every_n_epochs=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # 训练日志（TensorBoard）
    tb_logger = TensorBoardLogger(save_dir=output_dir, name="tensorboard", version=None)

    # 4. 启动 Trainer
    use_gpu = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if use_gpu else 0
    devices = args.devices if args.devices is not None else (n_gpus or 1)
    use_ddp = use_gpu and devices > 1
    precision = "16-mixed" if use_gpu else "32"
    strategy = "ddp" if use_ddp else "auto"
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=devices,
        strategy=strategy,
        precision=precision,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10 if args.single_region_dir else 50, # 单图模式打印勤一点
        val_check_interval=1.0 if args.single_region_dir else 0.25, # 单图每epoch验证
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    
    if use_ddp:
        print(f"使用 {devices} 张 GPU 进行 DDP 分布式训练")

    print("\n🚀 开始训练...")
    trainer.fit(model, train_loader, val_loader)
    print(f"\n✅ 训练完成。最佳权重已保存在: {ckpt_dir}")
    print(f"   训练曲线: tensorboard --logdir {os.path.join(output_dir, 'tensorboard')} --port 6006")

# ==========================================
# 4. 命令行参数
# ==========================================
def parse_args():
    p = argparse.ArgumentParser(
        description="SAMRoute finetune — 多图批量正式训练 (支持 DDP、TF32、cached features)",
        epilog="单图过拟合测试: python finetune_demo.py --single_region_dir path/to/region --epochs 100"
    )
    p.add_argument("--data_root", default="Gen_dataset_V2/Gen_dataset", help="数据集根目录")
    
    # 🎯 单图测试开关
    p.add_argument("--single_region_dir", type=str, default=None, 
                   help="如果提供单个图像的文件夹路径（包含 .tif 和 .png），则无视 data_root，只训练这一个区域！")
                   
    p.add_argument("--pretrained_ckpt", default="checkpoints/cityscale_vitb_512_e10.ckpt", help="预训练 SAM-Road 权重路径")
    p.add_argument("--output_dir", default="training_outputs/finetune_demo", help="输出目录")
    p.add_argument("--epochs", type=int, default=50, help="训练轮数")
    p.add_argument("--batch_size", type=int, default=32, help="batch size，cached 模式显存占用低可开 32~64")
    p.add_argument("--lr", type=float, default=None, help="学习率，默认 5e-4 (配合 batch 16)")
    p.add_argument("--val_fraction", type=float, default=0.1, help="验证集城市占比 (0~1)")
    p.add_argument("--samples_per_region", type=int, default=50, help="每区域每 epoch 采样数")
    p.add_argument("--road_dilation_radius", type=int, default=3, help="归一化 mask 半径")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers，preload 时 4 足够")
    p.add_argument("--devices", type=int, default=1, help="GPU 数量，默认单卡训练")
    p.add_argument("--use_cached_features", action="store_true", help="使用预计算 samroad_feat_full_*.npy 跳过 Encoder")
    p.add_argument("--no_cached_features", action="store_true", help="关闭 cached features，始终跑 Encoder")
    p.add_argument("--no_preload", action="store_true", help="关闭 preload_to_ram")
    p.add_argument("--fast", action="store_true", help="快速模式：单卡 batch32 workers4，便于调试")
    # 参数扫描：ROAD_POS_WEIGHT / ROAD_DICE_WEIGHT / ROAD_THIN_BOOST
    p.add_argument("--road_pos_weight", type=float, default=None, help="BCE 正样本权重")
    p.add_argument("--road_dice_weight", type=float, default=None, help="Dice 损失权重，越大越强调细路")
    p.add_argument("--road_thin_boost", type=float, default=None, help="BCE 细路像素额外权重倍数 (默认 6.0)")
    p.add_argument("--run_name", type=str, default=None, help="参数扫描时指定，checkpoint 保存为 best_{run_name}.ckpt")
    # LoRA  encoder 微调
    p.add_argument("--encoder_lora", action="store_true", help="开启 LoRA 微调 Encoder")
    p.add_argument("--lora_rank", type=int, default=4, help="LoRA rank，仅 --encoder_lora 时生效")
    args = p.parse_args()
    args.preload_to_ram = not args.no_preload
    args.use_cached_features = args.use_cached_features and not args.no_cached_features
    if args.encoder_lora:
        args.use_cached_features = False  # LoRA 必须在线计算，不可用缓存
    if args.fast:
        args.devices = args.devices or 1
        args.batch_size = 32
        args.workers = 4
        args.use_cached_features = True
        print("⚠️ --fast: 单卡 batch=32 workers=4 use_cached_features=True")
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)