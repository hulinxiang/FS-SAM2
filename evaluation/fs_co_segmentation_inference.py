# -*- coding: utf-8 -*-
# FS-SAM2 Co-segmentation Inference (对齐 sam2/evaluation/co_segmentation_inference.py 的数据与评测流程)
import os, time, json, glob, sys, logging
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

# Hydra + sam2 数据管线（与 co_segmentation_inference.py 一致）
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra

from training.dataset.vos_dataset import VOSDataset
from training.dataset.coco_pair_dataset import COCOPairRawDataset
from training.dataset.vos_sampler import RandomUniformSampler
from training.utils.data_utils import collate_fn as collate_fn_impl

# FS-SAM2 模型
from peft import LoraConfig, get_peft_model
from sam2_pred import SAM2_pred

# ========== 配置 ==========
VAL_PAIRS_JSON = "/home/projects/u7633783/datasets/coco_pairs_val/pairs.json"   # 与 sam2 推理一致
SAM2_CFG_NAME = "configs/sam2.1_training/training_coco/coco_eval_2f.yaml"       # 与 sam2 推理一致（只用于数据变换）
CKPT_FS_SAM2 = "/home/projects/u7633783/sam2/logs/coco2017p/coseg_tiny/fold0.log/best_model.pt"  # 你的 FS-SAM2 ckpt
SAM2_IMAGE_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"                           # FS-SAM2 内部用到
SAM2_IMAGE_CKPT = "/home/projects/u7633783/sam2/checkpoints/sam2.1_hiera_tiny.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
NUM_WORKERS = 4

# ========== 日志 ==========
def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"fs_sam2_eval_{ts}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="w"); fh.setFormatter(fmt); fh.setLevel(logging.INFO); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); sh.setLevel(logging.INFO); logger.addHandler(sh)

    logging.info("="*60); logging.info("FS-SAM2 Co-segmentation Inference"); logging.info("="*60)
    return log_file

# ========== 评测指标（与sam2的脚本一致） ==========
def bin_iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    if pred.sum() == 0 and gt.sum() == 0: return 1.0
    inter = (pred & gt).sum().float()
    union = (pred | gt).sum().float().clamp_min(1)
    return (inter / union).item()

def dice_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    if pred.sum() == 0 and gt.sum() == 0: return 1.0
    inter = (pred & gt).sum().float()
    total = pred.sum().float() + gt.sum().float()
    return (2.0 * inter / total.clamp_min(1)).item()

def precision_recall(pred: torch.Tensor, gt: torch.Tensor) -> tuple:
    if pred.sum() == 0:
        precision = 1.0 if gt.sum() == 0 else 0.0
        recall = 1.0 if gt.sum() == 0 else 0.0
        return precision, recall
    tp = (pred & gt).sum().float()
    fp = (pred & ~gt).sum().float()
    fn = (~pred & gt).sum().float()
    precision = (tp / (tp + fp).clamp_min(1)).item()
    recall = (tp / (tp + fn).clamp_min(1)).item()
    return precision, recall

# ========== 构建与原 sam2 脚本一致的数据管线 ==========
def create_loader_from_hydra():
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_module("sam2", version_base="1.2")
    cfg = compose(config_name=SAM2_CFG_NAME)  # 仅用其中的 transform 与采样参数
    transforms = instantiate(cfg.vos.train_transforms, _recursive_=True)

    video_ds = COCOPairRawDataset(pairs_json_path=VAL_PAIRS_JSON)  # 与原脚本一致
    sampler = RandomUniformSampler(num_frames=cfg.scratch.num_frames, max_num_objects=cfg.scratch.max_num_objects)
    vos_ds = VOSDataset(
        transforms=transforms,
        training=False,
        video_dataset=video_ds,
        sampler=sampler,
        multiplier=1,
        always_target=True,
        target_segments_available=True,
    )
    collate = lambda batch: collate_fn_impl(batch, dict_key="all")
    loader = torch.utils.data.DataLoader(vos_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate)
    logging.info(f"Dataloader created: {len(vos_ds)} pairs, frames={cfg.scratch.num_frames}")
    return loader

# ========== 构建 FS-SAM2 + LoRA，并加载 ckpt ==========
def build_fs_sam2():
    model = SAM2_pred(model_cfg=SAM2_IMAGE_CFG, checkpoint=SAM2_IMAGE_CKPT)

    # LoRA 注入（与 train/test 一致）
    peft_encoder = get_peft_model(model.model.image_encoder, LoraConfig(inference_mode=False, r=4, lora_alpha=16, lora_dropout=0.1, target_modules=['qkv','proj'], bias="none"))
    model.model.image_encoder = peft_encoder
    peft_mem = get_peft_model(model.model.memory_attention, LoraConfig(inference_mode=False, r=32, lora_alpha=16, lora_dropout=0.1, target_modules=['q_proj','v_proj','k_proj','out_proj'], bias="none"))
    model.model.memory_attention = peft_mem
    peft_mem_enc = get_peft_model(model.model.memory_encoder, LoraConfig(inference_mode=False, r=32, lora_alpha=16, lora_dropout=0.1, target_modules=['out_proj'], bias="none"))
    model.model.memory_encoder = peft_mem_enc

    # 加载 FS-SAM2 训练权重
    obj = torch.load(CKPT_FS_SAM2, map_location="cpu")
    state_dict = obj['state_dict'] if 'state_dict' in obj else obj
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.info(f"Loaded FS-SAM2 ckpt: {CKPT_FS_SAM2} | missing={len(missing)}, unexpected={len(unexpected)}")

    model.eval().to(DEVICE)
    return model

# ========== 核心：将 VOS batch 转成 FS-SAM2 的“支持→查询”两步前向 ==========
@torch.no_grad()
def forward_fs_sam2_pair(fs_model: SAM2_pred, batch: dict):
    # VOS batch 结构：img_batch [T,B,C,H,W]、masks [T,O,H,W]；我们只用 T=2、B=1、O=1
    img_tbchw = batch['img_batch']            # [T,B,C,H,W]
    masks_tohw = batch['masks']               # [T,O,H,W]，O=1
    T, B = img_tbchw.shape[0], img_tbchw.shape[1]
    assert T >= 2 and B == 1, f"Expect T>=2 and B=1, got {img_tbchw.shape}"

    img0 = img_tbchw[0, 0]                    # [C,H,W]
    img1 = img_tbchw[1, 0]                    # [C,H,W]
    gt0  = masks_tohw[0, 0].float() > 0.5     # [H,W]
    gt1  = masks_tohw[1, 0].float() > 0.5     # [H,W]

    # 支持→建记忆
    mem = fs_model(img0.unsqueeze(0), mask_inputs=gt0.unsqueeze(0).to(img0.device), prev_out={})
    # 支持图预测（可选，与 sam2 对齐评估两帧）
    out0 = fs_model(img0.unsqueeze(0), prev_out=mem)
    # 查询图预测
    out1 = fs_model(img1.unsqueeze(0), prev_out=mem)

    # 取 high-res logits（train 里阈值 >0.0）
    logit0 = out0['logit_mask']               # [1,1,H,W]
    logit1 = out1['logit_mask']               # [1,1,H,W]
    pred0 = (logit0.squeeze(1) > 0.0).squeeze(0).detach().cpu().bool()
    pred1 = (logit1.squeeze(1) > 0.0).squeeze(0).detach().cpu().bool()

    return pred0, pred1, gt0.detach().cpu().bool(), gt1.detach().cpu().bool()

def main():
    save_dir = "/home/projects/u7633783/sam2_logs/fs_sam2_coseg_eval"
    setup_logging(save_dir)
    logging.info(f"Device: {DEVICE}")

    # 构建数据与模型（与 sam2 推理对齐）
    loader = create_loader_from_hydra()
    fs_model = build_fs_sam2()

    stats = {
        'total_pairs': 0, 'successful_pairs': 0, 'failed_pairs': [],
        'frame0': {'iou':[], 'dice':[], 'precision':[], 'recall':[]},
        'frame1': {'iou':[], 'dice':[], 'precision':[], 'recall':[]},
        'times': []
    }

    t0 = time.time()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if DEVICE.type=='cuda' else torch.no_grad():
        for i, batch in enumerate(loader):
            t_pair = time.time()
            try:
                # 转到设备
                batch = {k:(v.to(DEVICE) if torch.is_tensor(v) else v) for k,v in batch.items()}
                # 两步前向（支持→查询）
                pred0, pred1, gt0, gt1 = forward_fs_sam2_pair(fs_model, batch)

                # 逐帧统计
                for (p,g, key) in [(pred0,gt0,'frame0'), (pred1,gt1,'frame1')]:
                    iou = bin_iou(p,g); dice=dice_score(p,g); prec, rec = precision_recall(p,g)
                    stats[key]['iou'].append(iou); stats[key]['dice'].append(dice)
                    stats[key]['precision'].append(prec); stats[key]['recall'].append(rec)

                stats['successful_pairs'] += 1
                stats['times'].append(time.time()-t_pair)
            except Exception as e:
                logging.error(f"Pair {i} failed: {e}")
                stats['failed_pairs'].append(i)
            stats['total_pairs'] += 1

            if (i+1) % 100 == 0:
                recent = stats['frame1']['iou'][-100:] if len(stats['frame1']['iou'])>=100 else stats['frame1']['iou']
                logging.info(f"Processed {i+1} pairs, Recent Frame1 IoU mean={np.mean(recent):.4f}")

    dt = time.time() - t0
    logging.info(f"Done. Total={stats['total_pairs']} success={stats['successful_pairs']} fail={len(stats['failed_pairs'])}")
    logging.info(f"Avg time per pair={np.mean(stats['times']):.3f}s (total {dt:.2f}s)")

    for f in ['frame0','frame1']:
        for m in ['iou','dice','precision','recall']:
            arr = np.array(stats[f][m], dtype=float)
            logging.info(f"{f}.{m}: mean={arr.mean():.4f} std={arr.std():.4f} median={np.median(arr):.4f} "
                         f"min={arr.min():.4f} max={arr.max():.4f} N={len(arr)}")

if __name__ == "__main__":
    main()