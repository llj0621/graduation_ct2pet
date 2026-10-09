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

配置 `reference_root` 时，还记录生成 PET 对实际配对 PET NIfTI 的 MAE、MSE、PSNR、轴位 SSIM，逐病例 W&B 表和三方向的真实/生成/误差图。参考 PET 仅用于评价，裁剪到 `[0,5]`；像素指标固定 data range=5，单位沿用源 PET 数据。参考尺寸必须与解码结果一致，不把 latent 的 VAE 重建当作真实 PET。生成像素指标仅覆盖配置中固定数量的病例，test diffusion loss 覆盖完整测试集；不能把这几个病例的生成指标视为全测试集的生成指标。

默认 float32；CUDA 可以用 `--precision bfloat16` 或 `float16`。训练没有开始时配置错误会直接报出；数据合同、网络或优化器设置不匹配时拒绝续跑。

#### 参考图像与生成检查的三个配置

在 `configs/train_ct2pet.json` 中设置这三个字段：

| 配置 | 含义 |
| --- | --- |
| `reference_root` | 原始配对 PET 图像的根目录，评价时读取 `<reference_root>/<case>/pet.nii.gz`。设置为 JSON 的 `null` 时不读取原始图像、不进行 VAE 图像解码和图像指标评价。 |
| `validation_samples` | 每到 `sample_interval`，从验证集中选择固定数量的病例，从纯噪声完整采样 PET。设置为 `0` 时关闭这项生成检查。 |
| `test_samples` | 正式训练完成后，使用最佳 EMA 为固定数量的测试病例完整采样 PET。设置为 `0` 时关闭这项生成检查。 |

`reference_root` 不参与扩散训练 loss，也不是生成模型的输入条件。训练仍使用 `ct.pt/pet.pt`；`best.pt` 始终按验证集 `v_loss` 选择，不按 PET 图像指标选择。

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

这组设置仍会记录训练 v-loss、验证/测试 diffusion loss、验证/测试的一步 x0 误差，正常保存 `best.pt`、`last.pt` 和定期 checkpoint，并支持断点续训；关闭的是从纯噪声开始的完整 PET 生成、生成 latent 指标和图像评价。原始图像不再是训练依赖，可以减少采样、VAE 解码及 W&B 图片记录的耗时。多卡训练中，生成评价由主进程执行，其他卡等待；关闭生成检查也会减少这段等待时间。

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

### 验证代码

```powershell
uv run python -m unittest discover -s tests -v
```

测试覆盖 v-prediction 数学关系、CT 与 timestep 对输出的作用、两分支梯度、固定采样、数据分组隔离、变量尺寸分批、独立模态变换、严格输入检查、真实 PET 像素指标、训练恢复一致性，以及完全没有 PET 文件时的 CT-only 推理。

`test_ct2pet_distributed.py` 额外检查同形状全局分批、尾批覆盖、验证样本不重复及空分片，并用两个真实 Gloo 进程验证模型副本同步、按样本数加权的梯度、全卡验证计数、主进程错误传播和中断续训一致性。缺少可用 Gloo transport 的 Windows 构建会明确跳过两个多进程测试。

DDP 改造已在 Linux 容器中的 PyTorch `2.10.0+cpu` 通过全部 17 项测试，两个进程的模型副本相同，中断续训的 model/EMA 权重与连续训练逐元素相同。测试日志为 `outputs/ct2pet_ddp/linux_tests.log`。实际 `torchrun` 双进程四步检查也通过：仅创建一个 W&B 离线实验，日志不重复，checkpoint 保存两套 RNG 状态，多卡 checkpoint 可直接用于 CT-only 推理；记录见 `outputs/ct2pet_ddp/cli_verification.json` 和 `torchrun.log`。本机仅有一张 GPU，尚未实测多张 GPU 上的 NCCL 训练。

2026-10-09 本机检查：13 项测试通过；1,245 对真实 latent 数据预检查通过；5 例真实数据的小网络两步训练完成了离线 W&B 指标、图片、逐病例表记录，以及真实 PET VAE 解码和 CT-only 推理。另检查了完整默认网络的 float32、batch size=8 单步训练，在 RTX 4090 上峰值 tensor 显存约 4.79 GiB。检查产物保存在 `outputs/ct2pet_preflight/` 和 `outputs/ct2pet_smoke/`，这些短时检查不代表正式模型的生成质量。
