weight = None  # path to model weight (模型权重路径)
resume = False  # whether to resume training process (是否恢复训练进程)
evaluate = True  # evaluate after each epoch training process (每个epoch训练后进行评估)
test_only = False  # test process (仅测试模式)

seed = None  # train process will init a random seed and record (训练过程将初始化随机种子并记录)
save_path = "exp/default"  # save path for output files (输出文件保存路径)
num_worker = 16  # total worker in all gpu (所有GPU的总工作线程数)
batch_size = 16  # total batch size in all gpu (所有GPU的总批次大小)
gradient_accumulation_steps = 1  # total steps to accumulate gradients for (梯度累积的总步数)
batch_size_val = None  # auto adapt to bs 1 for each gpu (自动适配每个GPU的验证批次大小为1)
batch_size_test = None  # auto adapt to bs 1 for each gpu (自动适配每个GPU的测试批次大小为1)
epoch = 100  # total epoch, data loop = epoch // eval_epoch (总epoch数，数据循环次数 = epoch // eval_epoch)
eval_epoch = 100  # sche total eval & checkpoint epoch (评估和保存检查点的epoch间隔)
clip_grad = None  # disable with None, enable with a float (None表示禁用梯度裁剪，设置为浮点数表示启用)

sync_bn = False
enable_amp = False
amp_dtype = "bfloat16"
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False

enable_wandb = False
wandb_project = "pointcept"  # custom your project name e.g. Sonata, PTv3
wandb_key = None  # wandb token, default is None. If None, login with `wandb login` in your terminal

mix_prob = 0
param_dicts = None  # example: param_dicts = [dict(keyword="block", lr_scale=0.1)]

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# Trainer
train = dict(type="DefaultTrainer")

# Tester
test = dict(type="SemSegTester", verbose=True)
