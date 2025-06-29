# VAE+Diffusion RL 使用指南

本指南展示如何使用现有的 `run_experiment.py` 来运行 VAE+Diffusion RL 训练。

## 文件结构

- `train.py` - 新的训练脚本，专门用于 VAE+Diffusion RL
- `run_experiment.py` - 现有的实验管理脚本（无需修改）
- `exp_specs/vae_diffusion_rl_example.yaml` - VAE+Diffusion RL 配置示例
- `diffuser/models/diffusion_rl.py` - VAE+Diffusion RL 模型实现
- `diffuser/datasets/sequence.py` - 包含 VAESequenceDataset
- `diffuser/utils/rl_trainer.py` - RL 训练器

## 使用方法

### 1. 基本使用

使用现有的 `run_experiment.py` 运行 VAE+Diffusion RL 训练：

```bash
python run_experiment.py -e exp_specs/vae_diffusion_rl_example.yaml -g 0
```

### 2. 配置文件说明

配置文件 `exp_specs/vae_diffusion_rl_example.yaml` 的关键部分：

```yaml
meta_data:
  script_path: "train.py"  # 使用我们的新训练脚本
  
constants:
  # 使用 VAE+Diffusion RL 模型
  diffusion: "models.DiffusionRL"
  loader: "datasets.VAESequenceDataset"
  
  # VAE 参数
  vae_latent_dim: 64
  trajectory_latent_dim: 128
  vae_weight: 1.0
  
  # RL 参数
  online_training: false  # 是否启用在线训练
  rl_learning_rate: 0.0003
  use_value_function: true
  
variables:
  online_training: [false, true]  # 测试离线和在线+离线训练
```

### 3. 训练模式

#### 离线训练（Offline Only）
```yaml
online_training: false
```
- 仅使用历史数据训练
- 使用标准的 `utils.Trainer`
- 训练 VAE 重构损失 + 扩散损失

#### 在线+离线训练（Online + Offline）
```yaml
online_training: true
```
- 使用历史数据 + 环境交互数据
- 使用 `utils.OfflineRLTrainer` (纯离线训练)
- 训练 VAE 损失 + 扩散损失 + 策略梯度损失

### 4. 关键参数调整

#### VAE 参数
- `vae_latent_dim`: 动作潜在空间维度（默认64）
- `trajectory_latent_dim`: 轨迹编码维度（默认128）
- `vae_weight`: VAE 损失权重（默认1.0）

#### RL 参数
- `rl_learning_rate`: RL 学习率（默认3e-4）
- `policy_gradient_weight`: 策略梯度损失权重（默认1.0）
- `value_function_weight`: 价值函数损失权重（默认0.5）
- `entropy_weight`: 熵正则化权重（默认0.01）

#### 训练调度
- `rl_warmup_steps`: RL 预热步数（默认10000）
- `rl_collect_freq`: 经验收集频率（默认1000）
- `rl_collect_steps`: 每次收集步数（默认100）
- `rl_update_freq`: RL 更新频率（默认100）

### 5. 监控指标

训练过程中会记录以下指标：
- `loss`: 总损失
- `diffusion_loss`: 扩散模型损失
- `vae_loss`: VAE 总损失
- `vae_recon_loss`: VAE 重构损失
- `vae_kl_loss`: VAE KL 散度损失
- `policy_loss`: 策略损失（仅在线训练）
- `value_loss`: 价值函数损失（仅在线训练）

### 6. 自定义配置

要创建自己的配置文件，复制 `exp_specs/vae_diffusion_rl_example.yaml` 并修改：

```yaml
# 复制示例配置
cp exp_specs/vae_diffusion_rl_example.yaml exp_specs/my_vae_rl_config.yaml

# 编辑配置
vim exp_specs/my_vae_rl_config.yaml

# 运行实验
python run_experiment.py -e exp_specs/my_vae_rl_config.yaml -g 0
```

### 7. 多GPU 训练

```bash
# GPU 0
python run_experiment.py -e exp_specs/vae_diffusion_rl_example.yaml -g 0

# GPU 1
python run_experiment.py -e exp_specs/vae_diffusion_rl_example.yaml -g 1
```

### 8. 实验变量

配置文件支持变量扫描：

```yaml
variables:
  seed: [100, 200, 300]  # 多个随机种子
  vae_latent_dim: [32, 64, 128]  # 不同的潜在维度
  online_training: [false, true]  # 离线 vs 在线+离线
  rl_learning_rate: [1e-4, 3e-4, 1e-3]  # 不同学习率
```

这将自动生成所有组合的实验。

### 9. 日志和检查点

- 实验日志保存在 `logs/` 目录
- 检查点保存在对应的实验目录下
- 支持断点续训（`continue_training: True`）

### 10. 评估

配置评估器来定期评估模型性能：

```yaml
evaluator: "utils.MADEvaluator"
eval_freq: 100000  # 每10万步评估一次
num_eval: 10  # 评估10个episode
```

## 故障排除

1. **导入错误**: 确保所有新文件都在正确位置
2. **CUDA 内存不足**: 减少 `batch_size` 或 `vae_latent_dim`
3. **环境不兼容**: 离线训练模式不需要环境，在线训练需要
4. **收敛问题**: 调整学习率和权重平衡

## 示例命令

```bash
# 快速测试（离线训练）
python run_experiment.py -e exp_specs/vae_diffusion_rl_example.yaml -g 0

# 完整实验（包含在线训练的所有变量组合）
python run_experiment.py -e exp_specs/vae_diffusion_rl_example.yaml -g 0
```

这样您就可以完全复用现有的 `run_experiment.py`，同时使用新的 VAE+Diffusion RL 训练功能！ 