import argparse
import yaml

def get_opts():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--root_dir', type=str, required=True, help='directory of images')
    parser.add_argument('--flow_dir', type=str, default=None, help='directory of flow npy files')
    parser.add_argument('--log_save_path', type=str, default='logs', help='save log to')
    parser.add_argument('--model_save_path', type=str, default='ckpts', help='save checkpoint to')
    parser.add_argument('--exp_name', type=str, default='exp', help='experiment name')

    # Dimensions
    parser.add_argument('--img_wh', nargs="+", type=int, default=[854, 480], help='resolution (w, h)')
    parser.add_argument('--canonical_wh', nargs="+", type=int, default=[1000, 520], help='canonical resolution')

    # Training
    parser.add_argument('--num_steps', type=int, default=10001)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--gpu', type=int, default=0)


    # 在 opt.py 中添加/确认这些参数
    parser.add_argument('--weight_path', type=str, default=None, help='pretrained model weight to load')
    parser.add_argument('--fps', type=int, default=15, help='video fps')
    
    # Optimizer & Scheduler (补充了缺失的部分)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr_scheduler', type=str, default='steplr', choices=['steplr', 'cosine', 'poly', 'exponential'])
    parser.add_argument('--decay_step', nargs='+', type=int, default=[2500, 5000, 7500])
    parser.add_argument('--decay_gamma', type=float, default=0.5)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--warmup_multiplier', type=float, default=1.0)
    parser.add_argument('--num_epochs', type=int, default=10001) # 用于 scheduler 计算
    parser.add_argument('--poly_exp', type=float, default=0.9)   # 用于 poly scheduler

    # Weights
    parser.add_argument('--flow_loss', type=float, default=1.0)
    parser.add_argument('--flow_step', type=int, default=-1)

    # Annealing
    parser.add_argument('--annealed', action="store_true", default=True)
    parser.add_argument('--annealed_begin_step', type=int, default=4000)
    parser.add_argument('--annealed_step', type=int, default=4000)

    # Model Params (HashGrid)
    parser.add_argument('--deform_hash', action="store_true", default=True)
    parser.add_argument('--vid_hash', action="store_true", default=True)
    
    # Config
    parser.add_argument('--config', type=str, default=None, help='path to config file')

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Allow command line args to override config
        for k, v in config.items():
            if not hasattr(args, k) or getattr(args, k) is None: 
                setattr(args, k, v)
        
    return args