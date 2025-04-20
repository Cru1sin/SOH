from dataloader.XJTU_dataloader import XJTUdata
from Model.Model import PINN
import argparse
import os
import time
from datetime import datetime
import wandb

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_data(args):
    """
    加载数据
    :param args: 参数
    :return: 包含训练、验证和测试数据的字典
    """
    root = '/home/user/nss/BTM/CNN-PINN4QSOH/data/XJTU'  # mat文件所在目录
    data_loader = XJTUdata(root=root, args=args)
    
    # 读取所有batch的数据
    data = data_loader.read_one_batch()
    
    return {
        'train': data['train'],
        'valid': data['valid'],
        'test': data['test']
    }

def main():
    args = get_args()
    # 获取当前系统时间，格式化为字符串
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建保存结果的目录
    save_folder = os.path.join('/home/user/nss/BTM/CNN-PINN4QSOH/results of reviewer', 'XJTU', 'model_4.8')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    # 可以同时创建一个实验配置文件
    with open(os.path.join(save_folder, 'experiment_info.txt'), 'w') as f:
        f.write(f'Experiment started at: {current_time}\n')
    
    # 设置日志文件
    log_dir = os.path.join(save_folder, 'logging.txt')
    setattr(args, "save_folder", save_folder)
    setattr(args, "log_dir", log_dir)

    # 加载所有数据
    dataloader = load_data(args)
    
    # 初始化模型并训练
    if args.wandb:
        wandb.login()
        wandb.init(project=args.wandb_project_name, name=args.wandb_name)
    
    pinn = PINN(args)
    wandb.watch(pinn, log='all')
    pinn.Train(
        trainloader=dataloader['train'],
        validloader=dataloader['valid'],
        testloader=dataloader['test']
    )
    

def get_args():
    parser = argparse.ArgumentParser('XJTU数据集的超参数')
    
    # 数据相关参数
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', 
                       choices=['min-max', 'z-score'], help='normalization method')
    parser.add_argument('--minmax_range', type=tuple, default=(0, 1))
    parser.add_argument('--random_seed', type=int, default=2025)
    parser.add_argument('--sample_size', type=int, default=100)

    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--early_stop', type=int, default=20, help='early stopping patience')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epochs')
    parser.add_argument('--warmup_lr', type=float, default=0.002, help='warmup learning rate')
    parser.add_argument('--lr', type=float, default=0.01, help='base learning rate')
    parser.add_argument('--final_lr', type=float, default=0.0001, help='final learning rate')
    parser.add_argument('--lr_F', type=float, default=0.001, help='learning rate for F')

    # 模型相关参数
    parser.add_argument('--F_layers_num', type=int, default=3, help='number of layers in F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='hidden dimension of F')

    # 损失函数相关参数
    parser.add_argument('--alpha', type=float, default=0.7, 
                       help='weight for PDE loss (loss = l_data + alpha * l_PDE + beta * l_physics)')
    parser.add_argument('--beta', type=float, default=0.2, 
                       help='weight for physics loss (loss = l_data + alpha * l_PDE + beta * l_physics)')

    # 保存相关参数
    parser.add_argument('--save_folder', type=str, default='results/XJTU', 
                       help='folder to save results')
    parser.add_argument('--log_dir', type=str, default='logging.txt', 
                       help='log file path')
    
    parser.add_argument('--wandb', type=bool, default=True, 
                       help='use wandb to log')
    parser.add_argument('--wandb_project_name', type=str, default='CNN-PINN4QSOH', 
                       help='wandb project name')
    parser.add_argument('--wandb_name', type=str, default='test_batch_1', 
                       help='wandb name')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()

