# VAE checkpoints

- `ct_vae_best.ckpt`: CT PVAE encoder, decoder and VQ codebook.
- `pet_vae_best.ckpt`: PET PVAE encoder, decoder and VQ codebook.

两个文件复制自 `/checkpoints`，完整性记录在 `manifest.json`。
运行时从 checkpoint 的 `hyper_parameters` 恢复模型配置；加载 encoder、decoder、codebook 时检查全部所需权重键。
