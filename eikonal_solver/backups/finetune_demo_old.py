"""
Finetune SAMRoute map_decoder on Gen_dataset_V2/Gen_dataset (multi-image batch training).

Optimized for multi-GPU (DDP), large batch, TF32, cached features, high-throughput data loading.
"""
import argparse
import os
import torch

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
        self.FREEZE_ENCODER = True   # 绝对冻结 SAM Encoder，只训练 Decoder
        self.FOCAL_LOSS = False
        self.TOPONET_VERSION = 'default'

        _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        _PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
        self.SAM_CKPT_PATH = os.path.join(_PROJECT_ROOT, "sam_road_repo", "sam_ckpts", "sam_vit_b_01ec64.pth")

        # 路由/损失配置
        self.ROUTE_COST_MODE = 'add'
        self.ROAD_POS_WEIGHT = 13.9
        self.ROAD_DICE_WEIGHT = 0.5
        self.ROAD_DUAL_TARGET = False

        self.ROUTE_LAMBDA_SEG = 1.0
        self.ROUTE_LAMBDA_DIST = 0.0
        self.BASE_LR = 5e-4  # 配合 batch_size=16，较 4 时适当放大
        self.ENCODER_LR_FACTOR = 0.1   # configure_optimizers 需此属性
        self.LR_MILESTONES = [150] 

    def get(self, key, default):
        return getattr(self, key, default)

# ==========================================
# 2. 训练流程 (使用 Lightning Trainer)
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

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"数据集根目录不存在: {data_root}")
    os.makedirs(ckpt_dir, exist_ok=True)

    config = TrainConfig()
    if args.lr is not None:
        config.BASE_LR = args.lr
    model = SAMRoute(config)

    # 手动冻结本阶段不参与 Loss 的参数，满足 DDP 校验，从而使用标准 ddp 而非龟速 find_unused_parameters
    if config.FREEZE_ENCODER:
        for p in model.image_encoder.parameters():
            p.requires_grad = False
    if config.ROUTE_LAMBDA_DIST == 0.0:
        model.cost_log_alpha.requires_grad = False
        model.cost_log_gamma.requires_grad = False
        model.eik_gate_logit.requires_grad = False
        for p in model.topo_net.parameters():
            p.requires_grad = False

    # 1. 安全加载预训练权重
    if os.path.isfile(PRETRAINED_CKPT):
        print(f"加载预训练权重: {PRETRAINED_CKPT}")
        ckpt = torch.load(PRETRAINED_CKPT, map_location='cpu', weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        clean_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=False)
    else:
        print(f"⚠️ 未找到预训练权重 {PRETRAINED_CKPT}，从随机 Decoder 开始训练。")

    # 2. 构建 DataLoader
    # use_cached_features=True 时跳过 Encoder 计算，用预存 .npy 特征，可 10x+ 提速
    print("正在构建数据集并加载到内存...")
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

    # 3. 配置 Lightning 回调（仅保留 best + last，固定文件名避免累积）
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best',                    # 固定名 best.ckpt，覆盖旧文件
        save_top_k=1,
        monitor='val_seg_loss',
        mode='min',
        save_last=True,
        every_n_epochs=1,                   # 每 epoch 检查一次
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
    # 已手动冻结 encoder/eikonal/topo，可用标准 ddp（find_unused 是性能杀手）
    strategy = "ddp" if use_ddp else "auto"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=devices,
        strategy=strategy,
        precision=precision,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=50,           # 减少 step 级日志
        val_check_interval=0.25,       # 每 25% 的 train 做一次 val，分散开销
        enable_progress_bar=True,
        enable_model_summary=False,     # 跳过启动时的 model summary
    )
    if use_ddp:
        print(f"使用 {devices} 张 GPU 进行 DDP 分布式训练")

    print("\n🚀 开始正式微调训练...")
    trainer.fit(model, train_loader, val_loader)
    print(f"\n✅ 训练完成。最佳权重已保存在: {ckpt_dir}")
    print(f"   训练曲线: tensorboard --logdir {os.path.join(output_dir, 'tensorboard')} --port 6006")

# ==========================================
# 3. 命令行参数
# ==========================================
def parse_args():
    p = argparse.ArgumentParser(
        description="SAMRoute finetune — 多图批量正式训练 (支持 DDP、TF32、cached features)",
        epilog="示例: python finetune_demo.py --batch_size 16 --workers 8 --epochs 50"
    )
    p.add_argument("--data_root", default="Gen_dataset_V2/Gen_dataset", help="数据集根目录")
    p.add_argument("--pretrained_ckpt", default="checkpoints/cityscale_vitb_512_e10.ckpt", help="预训练 SAM-Road 权重路径")
    p.add_argument("--output_dir", default="training_outputs/finetune_demo", help="输出目录")
    p.add_argument("--epochs", type=int, default=50, help="训练轮数")
    p.add_argument("--batch_size", type=int, default=32, help="batch size，cached 模式显存占用低可开 32~64")
    p.add_argument("--lr", type=float, default=None, help="学习率，默认 5e-4 (配合 batch 16)")
    p.add_argument("--val_fraction", type=float, default=0.1, help="验证集城市占比 (0~1)")
    p.add_argument("--samples_per_region", type=int, default=50, help="每区域每 epoch 采样数")
    p.add_argument("--road_dilation_radius", type=int, default=3, help="归一化 mask 半径")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers，preload 时 4 足够")
    p.add_argument("--devices", type=int, default=None, help="GPU 数量，默认自动检测全部")
    p.add_argument("--use_cached_features", action="store_true", help="使用预计算 samroad_feat_full_*.npy 跳过 Encoder")
    p.add_argument("--no_cached_features", action="store_true", help="关闭 cached features，始终跑 Encoder")
    p.add_argument("--no_preload", action="store_true", help="关闭 preload_to_ram")
    p.add_argument("--fast", action="store_true", help="快速模式：单卡 batch32 workers4，便于调试")
    args = p.parse_args()
    args.preload_to_ram = not args.no_preload
    args.use_cached_features = args.use_cached_features and not args.no_cached_features
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