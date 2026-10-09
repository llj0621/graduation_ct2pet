# CT/PET VAE 编码与解码

## 目录

```text
graduation/
├── src/ct2pet_cervical_cancer/
│   ├── models/vae/       # CT/PET encoder、decoder、VQ codebook 与权重加载
│   ├── datasets/         # NIfTI、强度归一化、裁剪与 latent 变换
│   ├── inference/        # 公用滑窗编码和解码算法
│   ├── pipelines/        # 运行流程，CT/PET latent 解码共用 decode_latent.py
│   └── utils/            # 配置与项目路径
├── checkpoints/          # CT/PET VAE 权重
├── configs/              # 工作流配置
├── scripts/              # 命令行入口
└── outputs/               # 解码和预处理输出
```

## 安装

项目锁定了已验证的 PyTorch 2.10.0、MONAI 1.5.2 等依赖。PyTorch 默认来源是 CUDA 12.8，因此常规 `uv sync`、`uv run` 都会使用 CUDA 构建：

```powershell
uv sync
```

## CT/PET latent 重建

CT 与 PET 共用相同的单病例解码流程：读取一个病例的 latent，使用对应模态的 VQ codebook 和 decoder 还原影像，再从配对 NIfTI 继承空间坐标。默认输入为已配对 latent `ct.pt`、`pet.pt`，可分别用 `--case` 选择一例：

```powershell
uv run python scripts/decode_ct.py --case 101020072771
uv run python scripts/decode_pet.py --case 101020072771
```

CT 使用 `ct_vae_best.ckpt`，将 decoder 输出转换为 HU 并裁剪到 `[-1000,1000]`；PET 使用 `pet_vae_best.ckpt`，将输出换算并裁剪到 `[0,5]`。两者默认都使用对应 VQ codebook 量化 latent，输入输出空间尺寸必须与参考影像匹配。

输出位置：

```text
outputs/ct_latent_reconstruction/<病例号>/ct_reconstruction.nii.gz
outputs/pet_latent_reconstruction/<病例号>/pet_reconstruction.nii.gz
```

每个输出根目录另有 `ct_decode_manifest.json` 或 `pet_decode_manifest.json`。两条命令都接受相同的参数：

```text
--input-root --reference-root --output-root --checkpoint --latent-name
--device --gaussian-sigma --quantize/--no-quantize
--clip-output/--no-clip-output --case --overwrite
```

分别修改 `configs/decode_ct.json`、`configs/decode_pet.json` 中的输入、参考影像、权重和输出路径。二者配置字段相同。相对路径相对于项目根目录，因此可从其他当前目录执行。

如要解码 RF 生成的 PET latent，可指定对应输入目录和文件名，例如：

```powershell
uv run python scripts/decode_pet.py --input-root E:/graduation/0827_1/infer_teacher_t3 --latent-name pet_pred.pt --case 101021089644
```

若输出文件已存在，需加 `--overwrite`。

## 编码和预处理

`scripts/crop_ct_pet.py` 将原始 CT、PET 和 mask 对齐并按 mask 裁剪；`scripts/encode_latents.py` 将裁剪后的物理强度 NIfTI 编码为连续 `ct.pt`/`pet.pt`。CT 强度截断到 `[-1000,1000]` HU，PET 截断到 `[0,5]`，再分别缩放到 `[-1,1]`。latent 默认是 CPU float32 `[8,D,H,W]`，并记录病例 metadata 和总体 manifest。

`scripts/reconstruct_vae.py` 则从原始 CT/PET NIfTI 重新编码后再解码，可做 VAE round-trip 检查。它会同时处理两种模态；只检查某一例的 latent 重建时使用上面的 `decode_ct.py` 或 `decode_pet.py`。
