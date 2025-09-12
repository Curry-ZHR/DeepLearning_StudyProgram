import numpy as np
import pickle

# 直接放在同一目录下，然后读取
import pickle

def load_existing_weights():
    """加载现成的权重文件"""
    try:
        with open('sample_weight.pkl', 'rb') as f:
            network = pickle.load(f)
        
        print("权重文件加载成功!")
        print("网络结构:")
        for key, value in network.items():
            print(f"  {key}: {value.shape}")
        
        return network
    
    except FileNotFoundError:
        print("找不到 sample_weight.pkl 文件")
        return None
    except Exception as e:
        print(f"加载失败: {e}")
        return None

# 使用方法
network = load_existing_weights()