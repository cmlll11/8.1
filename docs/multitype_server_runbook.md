# 多类型实验服务器运行手册

本实验严格一次执行一个步骤：执行一条命令，等待输出，确认通过后再执行下一步。

## 步骤 1：检查环境

作用：确认 Python、PyTorch、CUDA 和 GPU 状态正确；不会启动长任务。

```bash
bash bash/00_check_environment.sh
```

## 步骤 2：训练模型和生成已知映射

作用：生成 clean/backdoor 分类器、UAP 和六类触发器 pair bundle。

```bash
nohup bash bash/09_train_multitype_assets.sh configs/multitype_feature_formal.yaml auto > outputs/multitype_assets.log 2>&1 & echo $! | tee outputs/multitype_assets.pid
```

`auto` 会选择显存占用最低的 GPU；也可以替换为 `cuda:0`。重复执行会复用已经完成的模型、UAP 和控制记录。

## 步骤 3：逐层提取特征

作用：固定模型和输入映射，提取 `stem` 到 `layer3` 的特征变化，并计算 AUROC、CKA 和变化统计。

```bash
nohup bash bash/10_multitype_observation.sh configs/multitype_feature_formal.yaml auto > outputs/multitype_observation.log 2>&1 & echo $! | tee outputs/multitype_observation.pid
```

## 步骤 4：逐个触发器和种子拟合

作用：在每层用候选映射拟合特征变化，逐候选写 checkpoint 和 `results.json`。

一次只启动一个命令；服务器有多个空闲 GPU 时，再由用户逐条启动其他组合。

```bash
nohup bash bash/11_multitype_fitting.sh configs/multitype_feature_formal.yaml auto 0 badnets > outputs/fit_seed0_badnets.log 2>&1 & echo $! | tee outputs/fit_seed0_badnets.pid
```

把 `0` 和 `badnets` 替换为需要的种子和触发器。每个候选的 checkpoint 在对应输出目录保存，SSH 断开后重复相同命令会续跑。

## 步骤 5：汇总

作用：计算每种触发器的 `K_fit`、`K_act`、UAP/Trigger bit 比值，并记录各自的首次不可区分层。

```bash
bash bash/12_multitype_report.sh configs/multitype_feature_formal.yaml
```

## 每一步回传的信息

用户每次只需要回传：

1. 完整命令；
2. 开始时间和结束时间；
3. 日志最后 20 行；
4. `status`、`run_id`、`valid` 或 `completed` 信息；
5. GPU 占用（若任务异常）；
6. 是否通过当前控制门槛。

不要因为 `tail -f` 没有新输出就终止任务；按 `Ctrl+C` 只会退出日志查看，不会停止 `nohup` 后台任务。

