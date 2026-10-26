# CT/PET VAE 与 CT 控制 PET 生成

## 目录

```text
graduation/
├── src/ct2pet_cervical_cancer/
│   ├── models/vae/       # CT/PET encoder、decoder、VQ codebook 与权重加载
│   ├── models/diffusion/ # CT 控制的 3D U-Net、v-prediction 与 DDPM scheduler
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

### 确认使用项目的 uv 环境

在包含 `pyproject.toml` 和 `uv.lock` 的项目根目录运行命令，使用 `uv run python`。uv 默认使用当前项目的 `.venv`，无需手动激活环境。可以先检查解释器、PyTorch 安装位置及版本：

```bash
uv run python -c "import sys,torch; print('Python:',sys.executable); print('Torch:',torch.__file__); print('Version:',torch.__version__)"
```

Linux/WSL 下，解释器应位于 `<项目目录>/.venv/bin/python`，PyTorch 应位于该 `.venv` 的 `site-packages`；Windows 下分别是 `<项目目录>\.venv\Scripts\python.exe` 和 `.venv\Lib\site-packages\torch`。以实际打印的路径确认使用了预期环境。详见 [uv 项目运行说明](https://docs.astral.sh/uv/concepts/projects/run/)。

多卡启动也使用 `uv run python -m torch.distributed.run`。`torchrun` 默认用启动它的 `sys.executable` 创建各卡进程，因此各卡使用同一个项目环境，无需分别激活。环境变量 `PYTHON_EXEC` 可以覆盖子进程解释器；下面的 Linux/WSL 启动示例先用 `unset PYTHON_EXEC` 清除该覆盖。

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

## CT→PET：CT 控制的 latent diffusion

本项目迁入 Leaper Phase B1 的纯 CT 控制 baseline。主 U-Net 对 PET latent 去噪，独立 CT encoder 的多级 skip 与 bottleneck 特征加到主分支上。主网络和控制分支一起训练；CT/PET VAE 保持固定，使用离线编码的连续 latent。当前没有文本、临床或检验条件。

默认设置：线性 DDPM schedule、1000 timesteps、v-prediction、Min-SNR gamma=5、AdamW、EMA，训练 300,000 optimizer steps。这里的 v-prediction 不是 Rectified Flow。迁移时显式设置 `time_dim=256`，修正原 B1 默认配置未把 timestep embedding 接入 U-Net 残差块的问题。网络包含必要的 self-attention，不保留 CT-only baseline 不使用的 cross-attention 参数。

训练输入使用当前病例目录布局：

```text
E:/graduation/latent_0820_LLJ/<case>/ct.pt
E:/graduation/latent_0820_LLJ/<case>/pet.pt
```

默认 `input_space=raw`，与本项目 `encode_latents.py` 一致。训练时分别使用 CT/PET checkpoint 的 codebook min/max 将 latent 转换到 diffusion 空间；transform、权重 SHA-256 和输入文件 SHA-256 都记录到配置和 checkpoint。已经 min/max 归一化的文件需显式设置 `--input-space codebook_minmax`；不能对它们再次归一化。其他标准化约定需先转换到这两种受支持空间。

不把裁剪数据 resize 回 `320³`：形如 `[8,40,40,10]`、`[8,40,40,20]` 的病例按相同形状分批；batch size 是上限，各组末批可能更小。

### 数据划分与检查

首次执行按固定 seed 创建 `outputs/ct2pet/split.json`，默认 train/validation/test 为约 80%/10%/10%。后续实验复用同一份 split，并检查病例完整性、文件哈希及分组是否跨集合。未提供患者映射时分组单位是病例目录，`split_grouping=case_id`，不能据此保证同一真实患者没有多次检查跨集合。

若已有患者信息，提供 JSON 映射，例如 `{"case001":"patient001","case002":"patient001"}`，使用 `--patient-map <path>`，并指定新的 split 路径；患者的全部病例会进入同一集合。也可以提供预先固定的 split JSON，其中 `records` 每行至少包含 `case`、`patient_id`、`split`，split 为 `train`、`validation` 或 `test`。

只检查输入、生成划分和数据合同，不训练、不初始化 W&B：

```powershell
uv run python scripts/train_ct2pet.py --dry-run
```

### 训练与 W&B

修改 `configs/train_ct2pet.json`。默认 W&B project 是 `ct2pet-cervical-cancer`，`wandb_entity=null` 使用登录账户的默认团队，`wandb_mode=online`。认证按 W&B 常规方式配置：

```powershell
uv run python -m wandb login --cloud
uv run python scripts/train_ct2pet.py
```

W&B 和本地 `metrics.jsonl` 记录训练 v-loss、学习率、梯度范数、实际 batch size，固定 seed 验证 v-loss 与一步 x0 误差、最终 test loss。每个生成验证周期使用固定病例和 seed，从纯噪声采样；记录完整生成的 latent MAE/MSE。验证 loss 选择 EMA 最佳 checkpoint；最终测试使用最佳 EMA，而不是最后一步权重。

配置 `reference_root` 时，还记录生成 PET 对实际配对 PET NIfTI 的 MAE、MSE、PSNR、轴位 SSIM，逐病例 W&B 表和三方向的真实/生成/误差图。默认参考 PET 仅用于评价；开启下文的 `paired_nifti` 图像 L1 时也用于监督。评价参考裁剪到 `[0,5]`；像素指标固定 data range=5，单位沿用源 PET 数据。参考尺寸必须与解码结果一致，不把 latent 的 VAE 重建当作真实 PET。生成像素指标仅覆盖配置中固定数量的病例，test diffusion loss 覆盖完整测试集；不能把这几个病例的生成指标视为全测试集的生成指标。

默认 float32；CUDA 可以用 `--precision bfloat16` 或 `float16`。训练没有开始时配置错误会直接报出；数据合同、网络或优化器设置不匹配时拒绝续跑。

#### 可选的图像域 L1

保留原来的 Min-SNR 加权 v-MSE，新增解码图像的局部 L1：

```text
z0_hat = scheduler.predict_x0(z_t, v_hat, t)
Y_hat = frozen_pet_decoder(pet_transform.inverse(z0_hat 的对应窗口))
image_l1 = mean(abs(Y_hat - Y_target))
total_loss = v_loss + image_l1_weight * mean(w(t) * 每病例 image_l1)
```

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `image_l1_weight` | `0.0` | 总损失中图像 L1 的系数；`0` 关闭该项，不加载训练用 decoder，维持原训练路径。 |
| `image_l1_target` | `"vae_reconstruction"` | 解码配对真实 `pet.pt` 的对应窗口作为目标。可选 `"paired_nifti"`，读取 `reference_root/<case>/pet.nii.gz` 的对应窗口作为目标。 |
| `image_l1_timestep_weighting` | `"linear"` | `w(t)=(T-t)/T`，t 为从 0 开始的噪声时间步；高噪声时降低图像约束。`"none"` 使用恒定权重 1。 |
| `image_l1_patch_size` | `80` | 解码图像窗口的边长，单位为图像体素，须为 PET VAE 缩放倍率 8 或 4 的正整数倍。 |

每例每个训练 step 随机选择一个对应 latent 窗口，窗口小于配置尺寸时使用该维度的全部 latent。预测和目标使用相同位置；验证/测试按病例 seed 固定窗口与噪声，方便比较。该项是**单步干净 latent 估计的局部图像误差**，不是完整 DDPM 生成图像的误差，也不是全体积均匀采样的精确估计；窗口边缘与完整滑窗融合的解码结果可能不同。

PET decoder 保持 eval、冻结参数，但预测端保留梯度；以 float32 解码，并通过 activation checkpointing 减少反传显存。预测不量化、不裁剪；`decode_quantize` 仅控制生成评价。优化器和 DDP 同步的仍是扩散网络参数，额外 decoder 在每卡各加载一份。开启 L1 会增加解码、反传的时间和显存开销。

`vae_reconstruction` 目标是 `D(z0)`，**不是原始 PET**，无需 `reference_root`，也无需病灶 mask。`paired_nifti` 才直接监督实际 PET：目标裁剪到 `[0,5]` 后除以 2.5 再减 1，转换到 decoder 的强度坐标；不裁剪预测。此时必须提供参考目录，所有病例的参考尺寸须为 latent 空间尺寸乘以 VAE 缩放倍率。图像还须与 latent 的预处理、裁剪位置和轴顺序一致，代码不自动配准或重采样，尺寸检查不能证明空间配准正确。

例如，在配置文件中加入或修改以下字段，并保留其他训练设置：

```json
{
  "image_l1_weight": 0.01,
  "image_l1_target": "vae_reconstruction",
  "image_l1_timestep_weighting": "linear",
  "image_l1_patch_size": 80,
  "reference_root": null,
  "validation_samples": 0,
  "test_samples": 0,
  "output_dir": "outputs/ct2pet/ct_control_image_l1",
  "run_name": "ct-control-image-l1"
}
```

`0.01` 是小权重消融的起点，不是已验证的最优值。也可用 CLI 覆盖配置：

```powershell
uv run python scripts/train_ct2pet.py --image-l1-weight 0.01 --output-dir outputs/ct2pet/ct_control_image_l1 --run-name ct-control-image-l1
```

多卡仍使用下文的 uv + DDP 入口，在命令尾部添加 `--image-l1-weight 0.01`、独立的 `--output-dir` 和 `--run-name`。使用实际 PET 作目标则添加 `--image-l1-target paired_nifti --reference-root <配对图像目录>`。

W&B 和 `metrics.jsonl` 在启用后分别记录 `train/val/test` 的 `v_loss`、`image_l1`（未加时间权重）、`image_l1_weighted`（加时间权重，尚未乘总系数）及 `total_loss`。多卡按病例数汇总这些指标。`best.pt` 继续按验证 `v_loss` 选择，以保持与 baseline 相同的选择规则；图像 L1 和真实生成效果需另外比较。

旧配置和旧 checkpoint 的缺失字段按默认值处理，关闭 L1 时仍可续跑。精确续跑要求 L1 系数、目标、窗口、时间权重一致，实际图像监督还要求参考目录及图像文件哈希一致；启用 `paired_nifti` 时启动检查会读取全部参考文件计算哈希。不能通过 `--resume` 把原来单一 L2 的实验改成复合损失。更换损失请创建独立实验，训练脚本不会自动把旧权重作为新实验的初始化。

#### 病灶 mask 混合采样（图像 L1）

通过 `lesion_mask_root` 指定独立的病灶 mask 目录。文件名与 latent 病例子目录对应：

```text
latent_0820_LLJ/101020072771/ct.pt
latent_0820_LLJ/101020072771/pet.pt
lesion_masks/101020072771.nii.gz
```

mask 为有限的 `0/1` 三维 NIfTI；当前数据尺寸为 `320×320×80`，应与编码 `pet.pt` 时使用的图像处于同一网格、裁剪位置及轴顺序。启动时检查尺寸等于 PET latent 空间尺寸乘以 decoder 倍率（8X 时是 `40×40×10` latent）。有 `reference_root` 时还会检查训练 mask 与对应 `pet.nii.gz` 的 affine 和尺寸；没有参考图像时只能检查尺寸，需自行确保空间一致。不会自动 resize 或重采样 mask。

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `lesion_mask_root` | `null` | mask 目录；按 `{case}.nii.gz` 查找，只要求训练集病例的 mask。 |
| `lesion_sampling_probability` | `0.0` | 每例训练时选择病灶引导窗口的概率，取值 `[0,1]`；例如 `0.5` 表示病灶引导和普通随机窗口混合。`0` 保持原采样及随机数路径，不读取 mask。 |

病灶分支先随机选取一个包含病灶体素的 latent 单元，再随机选择包含该单元的合法窗口，保证选中的图像块覆盖至少一个病灶体素，并保留周围组织。用块内“任一阳性体素”映射到 latent 单元，小病灶不会因缩小而丢失。普通分支仍随机选窗，也可能自然覆盖病灶。每例每步仍只解码一个图像块；`80³` 窗口在本数据的深度方向覆盖全部 80 层，主要在另外两个方向选位置。

空 mask 表示已确认无病灶，回退到普通随机窗口；缺失训练 mask 则报错，避免将未标注病例默认为阴性。若只有部分病例有标注，应先准备相应训练子集和独立划分，不能把未标注病例补成空 mask。验证/测试仍沿用原来的固定随机窗口，不读取 mask，完整 DDPM 生成和推理也不需要 mask。

该功能只修改图像 L1 的选块位置，网络仍接收完整 CT/PET latent，v-MSE 对完整 latent 计算；不增加病灶区域加权 L1，不修改网络结构。必须同时启用 `image_l1_weight > 0`，否则混合采样没有监督作用，启动会报错。

在现有训练配置中修改以下字段（将示例 mask 路径换为实际目录）：

```json
{
  "image_l1_weight": 0.01,
  "lesion_mask_root": "E:/graduation/lesion_masks",
  "lesion_sampling_probability": 0.5,
  "output_dir": "outputs/ct2pet/ct_control_mixed",
  "run_name": "ct-control-mixed"
}
```

先检查数据和 mask，不加载训练 decoder、不开始训练：

```powershell
uv run python scripts/train_ct2pet.py --image-l1-weight 0.01 --lesion-mask-root E:/graduation/lesion_masks --lesion-sampling-probability 0.5 --output-dir outputs/ct2pet/ct_control_mixed --run-name ct-control-mixed --dry-run
```

移除 `--dry-run` 即开始训练。Linux 双卡示例，数据路径改为服务器实际路径：

```bash
unset PYTHON_EXEC
CUDA_VISIBLE_DEVICES=0,1 uv run python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 scripts/train_ct2pet.py \
  --input-root /data/latent_0820_LLJ --reference-root /data/reference_pet \
  --batch-size 4 --image-l1-weight 0.01 \
  --lesion-mask-root /data/lesion_masks --lesion-sampling-probability 0.5 \
  --output-dir outputs/ct2pet/ct_control_mixed --run-name ct-control-mixed
```

W&B 和 JSONL 新增 `train/image_l1_lesion_sampled_fraction`（实际走病灶引导分支的病例比例，不是所有窗口的病灶命中率）和 `train/image_l1_empty_mask_fraction`。DDP 按实际样本数汇总；配置记录概率、训练 mask 数量及空 mask 数量。训练 mask 内容指纹写入数据合同；改变 mask、目录或概率后，需新建实验，不能作为原实验精确续跑。关闭该功能时旧 checkpoint 仍可按原配置续跑。

`test_ct2pet_lesion_sampling.py` 检查小病灶及边缘覆盖、混合比例、空 mask 回退、关闭功能时逐元素一致的结果及 RNG、验证不读取 mask、梯度、W&B 配置/日志、mask 内容变化检测和精确续跑。2026-10-25 本机回归共 36 项，32 项通过，4 项多进程检查因 Windows Gloo transport 限制跳过；真实 PET latent 和真实 decoder 配合合成 `320×320×80` 边缘 mask 的 CUDA 反传检查通过。记录见 `outputs/lesion_sampling_regression.log` 和 `outputs/lesion_sampling_gpu_check/result.json`；尚未在真实人工病灶 mask 上启动正式训练。

#### 参考图像与生成检查的三个配置

在 `configs/train_ct2pet.json` 中设置这三个字段：

| 配置 | 含义 |
| --- | --- |
| `reference_root` | 原始配对 PET 图像的根目录，评价时读取 `<reference_root>/<case>/pet.nii.gz`，也可供 `paired_nifti` 图像 L1 监督。设为 `null` 时不读取原始图像、不做真实 PET 图像指标评价；启用重建目标的 L1 时仍会解码训练窗口。 |
| `validation_samples` | 每到 `sample_interval`，从验证集中选择固定数量的病例，从纯噪声完整采样 PET。设置为 `0` 时关闭这项生成检查。 |
| `test_samples` | 正式训练完成后，使用最佳 EMA 为固定数量的测试病例完整采样 PET。设置为 `0` 时关闭这项生成检查。 |

默认 `reference_root` 不参与扩散训练 loss；只有开启 `image_l1_weight>0` 且设置 `image_l1_target="paired_nifti"` 时，它才用于监督。它始终不是生成模型的输入条件。训练仍使用 `ct.pt/pet.pt`；`best.pt` 始终按验证集 `v_loss` 选择，不按 PET 图像指标选择。

只把 `reference_root` 改为 `null`，保留非零 `validation_samples`、`test_samples` 时，仍会执行完整 DDPM 采样、保存生成的 `pet_pred.pt` 并记录生成 latent 的 MAE/MSE；只是没有 PET 图像 MAE/MSE、PSNR、SSIM 和真实/生成/误差对比图。提供图像目录时则额外加载固定的 PET VAE decoder，解码后与真实 PET 图像比较；参考图像尺寸必须与解码结果一致。生成指标仅覆盖选中的病例。

如果只需要训练及验证/测试 loss，希望关闭完整生成检查，将对应三个字段改为：

```json
{
  "reference_root": null,
  "validation_samples": 0,
  "test_samples": 0
}
```

这是配置片段，请保留完整配置中的其他字段。使用 JSON 的 `null`，不要写成字符串 `"null"`。

这组设置仍会记录训练 v-loss、验证/测试 diffusion loss、验证/测试的一步 x0 误差，正常保存 `best.pt`、`last.pt` 和定期 checkpoint，并支持断点续训；关闭的是从纯噪声开始的完整 PET 生成、生成 latent 指标和图像评价。原始图像不再是训练依赖，可以减少采样、生成评价的 VAE 解码及 W&B 图片记录的耗时；若启用了重建目标的图像 L1，其训练和验证窗口仍会解码。多卡训练中，生成评价由主进程执行，其他卡等待；关闭生成检查也会减少这段等待时间。

若 `reference_root` 非 `null`，启动时仍会检查目录是否存在，即使生成病例数为 `0`。完全不需要原始图像时应将它设为 `null`。

```powershell
# 离线 W&B，之后可用 wandb sync 上传该离线目录
uv run python scripts/train_ct2pet.py --wandb-mode offline

# 完整恢复模型、EMA、优化器、LR schedule、RNG 和 batch 位置
uv run python scripts/train_ct2pet.py --resume outputs/ct2pet/ct_control_baseline/last.pt
```

`best.pt`、`last.pt`、定期 checkpoint、配置、历史和样本保存在对应 `output_dir`。新实验使用新的 `--output-dir` 和 `--run-name`；续跑在原 output_dir 内执行。`--stop-after` 用于保留原 max_steps 学习率计划的短时检查。

### 多卡训练（DDP）

训练入口自动读取 `torchrun` 的 `RANK`、`LOCAL_RANK` 和 `WORLD_SIZE`，每张 GPU 对应一个进程；普通 `uv run python scripts/train_ct2pet.py` 继续使用单卡。`dist_backend=auto` 在支持 NCCL 的 CUDA 环境使用 NCCL，其余环境使用 Gloo，可用 `--dist-backend` 显式选择。后端选择遵循 [PyTorch 2.10 分布式说明](https://docs.pytorch.org/docs/2.10/distributed.html)。

以下命令在 Linux/WSL 的项目根目录执行。先用 `nvidia-smi` 查看 GPU 编号，然后通过 `CUDA_VISIBLE_DEVICES` 指定要使用的卡。例如，只使用编号为 **1、3** 的两张 GPU，同时保证使用当前项目的 uv 环境：

```bash
nvidia-smi
unset PYTHON_EXEC
CUDA_VISIBLE_DEVICES=1,3 uv run python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 scripts/train_ct2pet.py \
  --device cuda --batch-size 4 \
  --output-dir outputs/ct2pet/ct_control_ddp2 --run-name ct-control-ddp2
```

进程内会把所选的 GPU 1、3 重新编号为逻辑 `cuda:0`、`cuda:1`，由 `LOCAL_RANK` 自动分配。`--nproc-per-node` 必须等于选择的卡数；`--device` 保持 `cuda`，不指定具体编号。`CUDA_VISIBLE_DEVICES` 和 `PYTHON_EXEC` 是启动进程的环境设置，不填写到训练 JSON 中。

使用编号为 0、1、2、3 的四张 GPU：

```bash
unset PYTHON_EXEC
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 scripts/train_ct2pet.py \
  --device cuda --batch-size 2 \
  --output-dir outputs/ct2pet/ct_control_ddp4 --run-name ct-control-ddp4
```

`batch_size` 是**每卡**上限；双卡各 4 例或四卡各 2 例时，总 batch 上限均为 8。保持原单卡总 batch=8 时，无需因卡数变化自动增大学习率。`max_steps` 仍按所有卡共同完成的 optimizer update 计数。服务器上需先将配置中的 `E:/...` 数据路径改为服务器实际路径；没有原始参考图像可设 `reference_root=null`。每个新实验使用独立的 `output_dir`。

训练 sampler 在同形状的全局 batch 内分配各卡样本，保证每卡训练步数相同，保留不同裁剪尺寸。尾批有足够样本时不补齐，并按各卡样本数对 loss/梯度加权；尾批少于卡数时仅重复到每卡至少一例。因此少量尾批可能包含重复病例。验证和测试不补齐、不重复，允许某卡分片为空，最终按病例数汇总指标。`train/batch_size` 记录实际总 batch，配置另记录每卡 batch 上限及 world size。

只有 rank 0 创建划分、初始化 W&B、写入配置/JSONL/checkpoint，以及执行生成 PET 图像评价；其他卡同步等待生成评价完成。所有卡共同计算验证/测试 diffusion loss，使用汇总后的验证 loss 选择最佳 EMA。主进程的保存、日志或生成评价错误会通知其他进程。

checkpoint 保存全部 rank 的 Python、NumPy、Torch 和当前 CUDA 设备 RNG 状态，同时保留不含 `module.` 前缀的模型权重，可直接使用现有单卡推理入口。续跑要求保持原 world size、device 类型、通信后端、每卡 batch size 和优化设置；旧单卡 checkpoint 仍可单卡续跑，但不能直接切换为多卡续跑。

```bash
unset PYTHON_EXEC
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 scripts/train_ct2pet.py \
  --device cuda --batch-size 2 --output-dir outputs/ct2pet/ct_control_ddp4 \
  --resume outputs/ct2pet/ct_control_ddp4/last.pt
```

本机 Windows 的 PyTorch `2.10.0+cu128` 虽报告 Gloo 可用，但实际创建通信设备报 `unsupported gloo device`，且不提供 NCCL；这个构建无法运行 DDP。使用 Linux/WSL CUDA 环境运行正式多卡训练；启动入口会明确提示缺失的 Gloo transport，不会退化成多个独立训练进程。

### 仅从 CT 生成 PET

推理只读取 `ct.pt`，不读取真实 PET latent。默认使用训练的最佳 EMA，从纯高斯噪声执行 DDPM 采样，输出原始 VAE 空间的 `pet_pred.pt`；解码时从对应 CT NIfTI 继承空间坐标。

```powershell
uv run python scripts/infer_ct2pet.py --case 101020072771
uv run python scripts/infer_ct2pet.py --case 101020072771 --no-decode
```

默认 `quantize=false`，与合作者 B1 连续 latent 解码路径一致；可加 `--quantize` 做量化解码对照。生成 NIfTI 默认裁剪到 `[0,5]`，训练生成评价则使用未裁剪的 decoder 输出。独立使用 VAE 解码脚本时需对应设置：

```powershell
uv run python scripts/decode_pet.py --input-root outputs/ct2pet/predictions --latent-name pet_pred.pt --case 101020072771 --no-quantize
```

### 独立病灶评价（完整测试集）

入口为 `scripts/evaluate_lesions.py`，配置为 `configs/evaluate_lesions.json`，也可运行 `uv run ct2pet-evaluate-lesions`。先执行 `uv sync` 安装新增的 SciPy 依赖。脚本独立于训练运行，不改变训练 loss 或 checkpoint 选择，不自动上传 W&B。

默认从训练保存的 `split.json` 中读取 **全部 test 病例**，只读取所选 split。文件对应关系如下：

```text
reference_root/<case>/pet.nii.gz          # 原始物理强度 PET
mask_root/<case>.nii.gz                  # 人工核对的 0/1 病灶 mask
prediction_root/<case>/pet_prediction.nii.gz
```

读取已有生成 PET，无需 GPU、latent 或模型权重：

```bash
uv run python scripts/evaluate_lesions.py \
  --split-manifest /data/1011_exp/split.json \
  --reference-root /data/reference_data --mask-root /data/masks_backup \
  --prediction-root /data/predictions/mixed_120k \
  --output-dir outputs/lesion_eval/mixed_120k
```

训练结束时默认只生成 16 个测试病例；直接指向这些结果不能当作完整测试集。脚本遇到缺失病例默认报错。只有明确添加 `--allow-missing` 才评价现有子集，报告会列出遗漏病例并标记 `complete_split=false`；比较两组时必须核对 `protocol.case_names` 一致。

若还没有完整测试集的生成 PET，可指定 checkpoint，脚本会先对测试集逐例仅从 CT 生成，再评价。模型和 VAE decoder 只加载一次；默认单卡，生成设置沿用 checkpoint 中的调度、latent 变换和量化设置，输出不裁剪：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/evaluate_lesions.py \
  --split-manifest /data/1011_exp/split.json \
  --reference-root /data/reference_data --mask-root /data/masks_backup \
  --checkpoint /data/mixed_run/step_000120000.pt \
  --input-root /data/latent_data --pet-checkpoint checkpoints/pet_vae_best.ckpt \
  --prediction-root outputs/lesion_eval/mixed_120k/predictions \
  --output-dir outputs/lesion_eval/mixed_120k/metrics --device cuda
```

生成需要同网格的 `reference_root/<case>/ct.nii.gz` 用于空间信息；参考 PET 和人工 mask 不进入采样。每例保存 `lesion_generation_metadata.json`，记录 checkpoint/CT/预测文件 SHA256、checkpoint 步数、seed、权重和解码设置。中断后使用同一命令可复用匹配的预测；改变 checkpoint、seed 或 CT 后会拒绝混用。覆盖已有生成结果需显式 `--regenerate`；覆盖统计报告用 `--overwrite`，两者独立。比较随机与混合采样时，换成相同步数的另一组 checkpoint，并使用不同的 prediction/output 目录。

评价输出：

| 文件 | 内容 |
| --- | --- |
| `per_case.csv` | 每病例病灶 MAE/MSE/RMSE、偏差、均值/峰值、整体 MAE/MSE、病灶/热点计数、强度超界比例和文件指纹。 |
| `per_lesion.csv` | 每个真实病灶连通域的体积、强度误差、世界坐标质心、匹配热点 IoU 与匹配后的质心距离。 |
| `per_hotspot.csv` | 预测高摄取连通域的体积、强度、位置及匹配标记，可用于人工核查未匹配热点。 |
| `summary.json` | 汇总指标、完整性、实际病例列表、评价协议和缺失病例。 |
| `results.json` | 包含逐病例、逐病灶、逐热点和汇总的完整数值结果。 |

主指标建议先看 `case_macro_lesion_mae`、`case_macro_lesion_bias`、`voxel_micro_lesion_mae`、`lesion_macro_centroid_distance_mm` 和病灶/热点匹配计数。病例宏平均对每个阳性病例等权，体素微平均按病灶体素数加权，病灶宏平均对每个真实连通域等权。空 mask 病例不参与病灶误差平均，但参与热点统计；无定义的比例和误差记录为 `null`，不填零。质心距离只统计已匹配病灶，需与漏灶数同时看。

强度误差默认对参考 PET 截断到 `[0,5]`，预测保持文件中的物理强度，以衔接现有生成指标；`raw_*` 字段另对未截断的参考 PET 计算误差/强度统计，并记录病灶中参考值超过 5 的比例。也可在 JSON 中设 `reference_clip_max=null` 关闭上界截断。旧 VAE 在编码前已截断到 5，不能用新评价恢复已丢失的高 SUV 信息。输入必须是物理强度 NIfTI，不能传入 `[-1,1]` 的 decoder 数据。已被上游截断的预测也无法由脚本还原。

热点是 **预测 PET > `hotspot_threshold`** 的连通域，默认阈值 `2.5`；真实病灶直接取人工 mask，不再以 SUV 阈值裁剪。默认 26 邻接（`connectivity=3`），一对一匹配采用 IoU≥`match_iou`（默认 `0.1`），先最大化有效匹配数量、再最大化 IoU；一个大热点不能重复算作检出多个病灶。`min_hotspot_volume_mm3` 默认 `0`（不过滤），可显式指定固定的最小热点体积。

**这些热点阈值/匹配规则是探索性评价协议，不是已验证的临床判定标准。** 未匹配热点可能包含膀胱等生理摄取，不等价于肿瘤假阳性；`hotspot_lesion_precision`、`hotspot_mask_dice` 等也不等价于语义分割性能。可用 `--evaluation-mask-root` 提供同样按 `{case}.nii.gz` 命名的有效评价区（1=保留），统一排除预先定义的生理区域；它只限制热点搜索，不改变病灶强度误差或整体误差。有效区必须保留所有人工病灶体素。阈值、IoU 和排除规则应在验证集确定后固定，再用于测试，不能按每组测试结果单独调整。默认截断协议要求热点阈值小于 5，避免将训练已截断的数据直接用于 `SUV>5` 检出。

预测、参考 PET、病灶 mask 及有效评价区均检查三维尺寸和 affine，不自动配准/重采样。空间单位应为 mm；NIfTI 未声明单位时按 mm 解释并在逐病例结果中记录 `spatial_units=unknown`。本脚本不计算“mask 内 SSIM”，避免把清零背景后的 SSIM 误作病灶结构分数。

2026-10-26 检查：评价新增 12 项测试通过，覆盖强度截断/原始口径、精确匹配/位移/漏灶/额外热点、微平均与宏平均、空 mask、网格检查、完整测试集要求、预测来源缓存校验和输入保护。整体 48 项测试中 44 项通过，4 项多进程测试因本机 Windows Gloo transport 限制跳过。另使用一例真实 `320×320×80` 人工 mask 和旧小网络生成 PET 跑通命令行与全部报表，记录于 `outputs/lesion_evaluation_smoke/`；其中 split 为单例读写检查夹具，不是研究正式测试划分，指标不能用于评价正式模型。

### 验证代码

```powershell
uv run python -m unittest discover -s tests -v
```

测试覆盖 v-prediction 数学关系、CT 与 timestep 对输出的作用、两分支梯度、固定采样、数据分组隔离、变量尺寸分批、独立模态变换、严格输入检查、真实 PET 像素指标、训练恢复一致性，以及完全没有 PET 文件时的 CT-only 推理。

`test_ct2pet_image_loss.py` 检查 decoder 反归一化、时间权重、图像 L1 到扩散两分支的梯度、VAE 冻结、原始 PET 窗口位置与强度坐标、分项日志/W&B 调用，以及新旧损失配置的续训一致性和兼容性。

`test_ct2pet_distributed.py` 额外检查同形状全局分批、尾批覆盖、验证样本不重复及空分片，并用两个真实 Gloo 进程验证模型副本同步、按样本数加权的梯度、全卡验证计数、主进程错误传播和中断续训一致性，包含开启图像 L1 的回归检查。缺少可用 Gloo transport 的 Windows 构建会明确跳过三个多进程测试。

DDP 改造已在 Linux 容器中的 PyTorch `2.10.0+cpu` 通过全部 17 项测试，两个进程的模型副本相同，中断续训的 model/EMA 权重与连续训练逐元素相同。测试日志为 `outputs/ct2pet_ddp/linux_tests.log`。实际 `torchrun` 双进程四步检查也通过：仅创建一个 W&B 离线实验，日志不重复，checkpoint 保存两套 RNG 状态，多卡 checkpoint 可直接用于 CT-only 推理；记录见 `outputs/ct2pet_ddp/cli_verification.json` 和 `torchrun.log`。本机仅有一张 GPU，尚未实测多张 GPU 上的 NCCL 训练。

2026-10-09 本机检查：13 项测试通过；1,245 对真实 latent 数据预检查通过；5 例真实数据的小网络两步训练完成了离线 W&B 指标、图片、逐病例表记录，以及真实 PET VAE 解码和 CT-only 推理。另检查了完整默认网络的 float32、batch size=8 单步训练，在 RTX 4090 上峰值 tensor 显存约 4.79 GiB。检查产物保存在 `outputs/ct2pet_preflight/` 和 `outputs/ct2pet_smoke/`，这些短时检查不代表正式模型的生成质量。

2026-10-11 图像 L1 检查：26 项测试中 23 项通过，3 项多进程测试因本机 Gloo transport 不可用跳过。使用 5 例真实 latent 和真实 PET VAE，在 GPU 上完成小扩散网络的重建目标两步训练、原始 PET 目标一步训练，以及 bfloat16 输入的图像 L1 反传检查；训练、验证、测试分项指标均正常，VAE 参数保持冻结。W&B 新增字段通过 SDK 调用替身验证；本机受限环境的 W&B 离线服务未完成启动，因此本次 GPU 检查使用 `wandb_mode=disabled`，未验证实际 W&B 实验文件或在线上传。检查产物在 `outputs/ct2pet_image_l1_checks/`，不代表病灶质量已改善。
