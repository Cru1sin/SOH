from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from scipy import stats
from scipy.stats import kurtosis, skew
from scipy.stats.qmc import LatinHypercube
import pickle

class MatDataLoader:
    def __init__(self, args):
        self.normalization = True
        self.normalization_method = args.normalization_method  # min-max, z-score
        self.args = args
        self.max_capacity = 2.0  # 默认最大容量
        self.sample_size = args.sample_size  # 拉丁超立方采样大小

    def _calculate_statistics(self, data_sequence):
        """
        计算时间序列的统计特征
        :param data_sequence: 时间序列数据
        :return: 统计特征数组
        """
        if len(data_sequence) == 0:
            return np.zeros(7)  # 返回7个特征的零数组
        
        # 计算基本统计量
        mean = np.mean(data_sequence)
        std = np.std(data_sequence)
        kurt = kurtosis(data_sequence)
        skewness = skew(data_sequence)
        
        # 计算累积电荷（电流的积分）
        cumulative_charge = np.trapz(data_sequence)
        
        # 计算曲线斜率（使用线性回归）
        if len(data_sequence) > 1:
            slope, _ = np.polyfit(range(len(data_sequence)), data_sequence, 1)
        else:
            slope = 0
            
        # 计算曲线熵
        hist, _ = np.histogram(data_sequence, bins=10, density=True)
        entropy = stats.entropy(hist)
        
        return np.array([mean, std, kurt, skewness, cumulative_charge, slope, entropy])

    def _latin_hypercube_sampling(self, data_dict, n_samples):
        """
        使用拉丁超立方采样对组合特征进行高维采样，保持时序性
        :param data_dict: 包含所有特征的字典
        :param n_samples: 采样数量
        :return: 采样后的数据字典
        """
        # 获取所有特征的长度
        lengths = [len(data) for data in data_dict.values()]
        min_length = min(lengths)
        
        if min_length <= n_samples:
            return data_dict
            
        # 创建组合特征矩阵，保持时序性
        combined_data = np.column_stack([data[:min_length] for data in data_dict.values()])
        
        # 对数据进行标准化，使每个维度都在[0,1]范围内
        data_min = np.min(combined_data, axis=0)
        data_max = np.max(combined_data, axis=0)
        normalized_data = (combined_data - data_min) / (data_max - data_min + 1e-10)
        
        # 使用拉丁超立方采样生成采样点
        sampler = LatinHypercube(d=combined_data.shape[1])
        sample_points = sampler.random(n_samples)
        
        # 为每个采样点找到最近的原始数据点
        sampled_indices = []
        for point in sample_points:
            # 计算采样点到所有数据点的距离
            distances = np.sum((normalized_data - point) ** 2, axis=1)
            # 找到最近的点的索引
            nearest_idx = np.argmin(distances)
            sampled_indices.append(nearest_idx)
        
        # 对索引进行排序以保持时序性
        sampled_indices = np.sort(sampled_indices)
        
        # 对每个特征进行采样
        sampled_data = {}
        for key, data in data_dict.items():
            sampled_data[key] = data[sampled_indices]
            
        return sampled_data

    def _parser_mat_data(self, battery_i_mat):
        '''
        解析mat文件数据
        :param battery_i_mat: shape:(1,len)
        :return: (data_norm_1, soh_loss_1), (data_norm_2, soh_loss_2)
        '''
        data = []
        label = []
        cycle_i = []
        
        for i in range(1, battery_i_mat.shape[1]):
            cycle_i_data = battery_i_mat[0,i]
            
            # 提取时间数据
            charge_time = cycle_i_data['relative_time_min'][0]
            # 提取其他特征
            current = cycle_i_data['current_A'][0]
            voltage = cycle_i_data['voltage_V'][0]
            temperature = cycle_i_data['temperature_C'][0]
            
            # 将所有特征组合在一起进行高维拉丁超立方采样
            data_dict = {
                'charge_time': charge_time,
                'current': current,
                'voltage': voltage,
                'temperature': temperature
            }
            
            sampled_data = self._latin_hypercube_sampling(data_dict, self.sample_size)
            
            # 组合每个时间步的数据
            for j in range(len(sampled_data['charge_time'])):
                # 计算历史统计特征
                current_stats = self._calculate_statistics(sampled_data['current'][:j+1])
                voltage_stats = self._calculate_statistics(sampled_data['voltage'][:j+1])
                relative_time = sampled_data['charge_time'][0] if j == 0 else sampled_data['charge_time'][j] - sampled_data['charge_time'][j-1] # 相对充电时间
                data_j = np.concatenate([
                    np.array([
                        relative_time,                      # 相对充电时间
                        sampled_data['charge_time'][j],     # 累计充电时间
                        i,                                  # 循环次数
                        sampled_data['current'][j],         # 电流
                        sampled_data['voltage'][j],         # 电压
                        float(sampled_data['temperature'][j])  # 温度
                    ]),
                    current_stats,          # 电流统计特征
                    voltage_stats           # 电压统计特征
                ])
                cycle_i.append(data_j)

            # 计算容量损失
            capacity_loss = cycle_i_data['capacity'][0] - battery_i_mat[0,i-1]['capacity'][0]
            label.append(capacity_loss)
            
            data.append(cycle_i)
            cycle_i = []

        data = np.array(data, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        
        # 使用dataloader中的归一化方式
        if self.normalization_method == 'min-max':
            data = 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1
        elif self.normalization_method == 'z-score':
            data = (data - np.mean(data)) / np.std(data)
        
        soh_loss = label / self.max_capacity

        data_norm_1 = data[0:-1]
        data_norm_2 = data[1:]
        soh_loss_1 = soh_loss[0:-1]
        soh_loss_2 = soh_loss[1:]

        return (data_norm_1, soh_loss_1), (data_norm_2, soh_loss_2)

    def load_mat_data(self, data_path):
        '''
        加载mat文件数据并处理
        :param data_path: mat文件路径
        :return: 处理后的数据
        '''
        print(f'Loading data from {data_path}...')
        mat = loadmat(data_path)
        battery = mat['battery']
        
        all_x1, all_y1 = [], []
        all_x2, all_y2 = [], []
        
        # 处理所有电池数据
        for battery_id in range(battery.shape[1]):
            print(f'Processing battery {battery_id + 1}...')
            battery_data = battery[0, battery_id][0]
            (x1, y1), (x2, y2) = self._parser_mat_data(battery_data)
            all_x1.append(x1)
            all_x2.append(x2)
            all_y1.append(y1)
            all_y2.append(y2)

        # 合并所有数据
        X1 = np.concatenate(all_x1, axis=0)
        X2 = np.concatenate(all_x2, axis=0)
        Y1 = np.concatenate(all_y1, axis=0)
        Y2 = np.concatenate(all_y2, axis=0)

        # 转换为tensor
        tensor_X1 = torch.from_numpy(X1).float()
        tensor_X2 = torch.from_numpy(X2).float()
        tensor_Y1 = torch.from_numpy(Y1).float().view(-1, 1)
        tensor_Y2 = torch.from_numpy(Y2).float().view(-1, 1)

        # 使用dataloader中的划分方式
        # 1. 首先划分训练集和测试集
        split = int(tensor_X1.shape[0] * 0.8)
        train_X1, test_X1 = tensor_X1[:split], tensor_X1[split:]
        train_X2, test_X2 = tensor_X2[:split], tensor_X2[split:]
        train_Y1, test_Y1 = tensor_Y1[:split], tensor_Y1[split:]
        train_Y2, test_Y2 = tensor_Y2[:split], tensor_Y2[split:]

        # 2. 从训练集中划分出验证集
        train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
            train_test_split(train_X1, train_X2, train_Y1, train_Y2, 
                           test_size=0.2, random_state=420)

        # 创建DataLoader
        train_loader = DataLoader(
            TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
            batch_size=self.args.batch_size,
            shuffle=True
        )
        valid_loader = DataLoader(
            TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
            batch_size=self.args.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(test_X1, test_X2, test_Y1, test_Y2),
            batch_size=self.args.batch_size,
            shuffle=False
        )

        # 创建完整数据的loader（用于其他目的）
        all_loader = DataLoader(
            TensorDataset(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2),
            batch_size=self.args.batch_size,
            shuffle=False
        )

        return {
            'train': train_loader,
            'valid': valid_loader,
            'test': test_loader,
            'all': all_loader
        }

class XJTUdata(MatDataLoader):
    def __init__(self, root, args):
        super(XJTUdata, self).__init__(args)
        self.root = root
        self.batch_names = ['batch-1','batch-2','batch-3','batch-4','batch-5','batch-6']
        self.max_capacity = 2.0
        self.feature_dir = os.path.join(root, 'handcraft-feature')
        self.raw_data_dir = os.path.join(root, 'charge')
        os.makedirs(self.feature_dir, exist_ok=True)
        print(f"Feature directory: {self.feature_dir}")
        print(f"Raw data directory: {self.raw_data_dir}")

    def _get_batch_feature_dir(self, batch):
        """获取批次对应的特征目录"""
        return os.path.join(self.feature_dir, f'handcraft-feature-{batch}')

    def _save_dataloader(self, batch, data):
        """保存处理后的DataLoader到文件"""
        batch_dir = self._get_batch_feature_dir(batch)
        os.makedirs(batch_dir, exist_ok=True)
        
        # 保存每个loader的数据
        for loader_name, loader in data.items():
            save_path = os.path.join(batch_dir, f'{loader_name}.pkl')
            # 保存tensors
            tensors = loader.dataset.tensors
            with open(save_path, 'wb') as f:
                pickle.dump(tensors, f)
            print(f"Saved {loader_name} data to {save_path}")

    def _load_dataloader(self, batch):
        """从文件加载DataLoader"""
        batch_dir = self._get_batch_feature_dir(batch)
        if not os.path.exists(batch_dir):
            return None
            
        data = {}
        for loader_name in ['train', 'valid', 'test', 'all']:
            save_path = os.path.join(batch_dir, f'{loader_name}.pkl')
            if not os.path.exists(save_path):
                return None
                
            with open(save_path, 'rb') as f:
                tensors = pickle.load(f)
                data[loader_name] = DataLoader(
                    TensorDataset(*tensors),
                    batch_size=self.args.batch_size,
                    shuffle=(loader_name != 'test' and loader_name != 'all')
                )
        return data

    def read_one_batch(self, batch='batch-1'):
        '''
        读取一个批次的mat文件，如果已处理过则直接加载
        :param batch: str or int
        :return: dict
        '''
        if isinstance(batch, int):
            batch = self.batch_names[batch]
        assert batch in self.batch_names, f'batch must be in {self.batch_names}'
        
        # 尝试加载已处理的数据
        cached_data = self._load_dataloader(batch)
        if cached_data is not None:
            print(f"Loaded cached data for {batch}")
            return cached_data
        
        # 从mat文件加载并处理
        print(f"Processing data for {batch}...")
        for file in os.listdir(self.raw_data_dir):
            if batch in file and file.endswith('.mat'):
                file_path = os.path.join(self.raw_data_dir, file)
                processed_data = self.load_mat_data(file_path)
                # 保存处理后的数据
                self._save_dataloader(batch, processed_data)
                return processed_data
        raise FileNotFoundError(f'No mat file found for batch {batch}')

    def read_all(self, specific_path_list=None):
        '''
        读取所有mat文件或指定的文件，优先使用缓存数据
        :param specific_path_list: 指定的文件路径列表
        :return: dict
        '''
        if specific_path_list is None:
            # 尝试加载所有批次的缓存数据
            all_data = None
            for batch in self.batch_names:
                batch_data = self._load_dataloader(batch)
                if batch_data is None:
                    print(f"No cached data found for {batch}, processing from mat files...")
                    break
                    
                if all_data is None:
                    all_data = batch_data
                else:
                    # 合并数据
                    for key in all_data:
                        all_data[key].dataset.tensors = tuple(
                            torch.cat((all_data[key].dataset.tensors[i], 
                                     batch_data[key].dataset.tensors[i])) 
                            for i in range(len(batch_data[key].dataset.tensors))
                        )
            
            if all_data is not None:
                print("Loaded all data from cache")
                return all_data
            
            # 如果缓存不存在，则从mat文件加载并处理
            print("Processing all data from mat files...")
            file_list = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.mat')]
            if not file_list:
                raise FileNotFoundError(f'No mat files found in {self.raw_data_dir}')
            
            # 读取第一个文件获取基础数据结构
            first_file = os.path.join(self.root, file_list[0])
            all_data = self.load_mat_data(first_file)
            
            # 处理剩余文件
            for file in file_list[1:]:
                file_path = os.path.join(self.root, file)
                data = self.load_mat_data(file_path)
                # 合并数据
                for key in all_data:
                    all_data[key].dataset.tensors = tuple(
                        torch.cat((all_data[key].dataset.tensors[i], 
                                 data[key].dataset.tensors[i])) 
                        for i in range(len(data[key].dataset.tensors))
                    )
            
            # 保存处理后的数据
            for batch in self.batch_names:
                self._save_dataloader(batch, all_data)
                
            return all_data
        else:
            return self.load_mat_data(specific_path_list[0])
    

if __name__ == '__main__':
    import argparse
    
    def get_args():
        parser = argparse.ArgumentParser('数据加载器参数')
        parser.add_argument('--normalization_method', type=str, default='z-score',
                          choices=['min-max', 'z-score'])
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--sample_size', type=int, default=100)
        return parser.parse_args()

    args = get_args()
    
    # 测试代码
    xjtu = XJTUdata(root='/home/user/nss/BTM/CNN-PINN4QSOH/data/XJTU', args=args)

    # 测试读取单个批次
    data = xjtu.read_one_batch('batch-1')
    print("\n单批次数据加载完成")
    print(f"训练集大小: {len(data['train'])}")
    print(f"验证集大小: {len(data['valid'])}")
    print(f"测试集大小: {len(data['test'])}")
    
    # 测试读取所有数据
    """all_data = xjtu.read_all()
    print("\n所有数据加载完成")
    print(f"训练集大小: {len(all_data['train'])}")
    print(f"验证集大小: {len(all_data['valid'])}")
    print(f"测试集大小: {len(all_data['test'])}")
    
    # 打印数据形状
    for x1, x2, y1, y2 in data['train']:
        print("\n数据形状:")
        print(f"x1: {x1.shape}")
        print(f"x2: {x2.shape}")
        print(f"y1: {y1.shape}")
        print(f"y2: {y2.shape}")
        break"""