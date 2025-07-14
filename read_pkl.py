import pickle
import re
import pandas as pd

def extract_metrics_from_log(log_file_path):
    """从训练日志中提取完整的metrics数据"""
    metrics_data = []
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # 匹配类似 "0: 11.3297 | bc_loss: 2.4255 | q_loss: 0.0331 | ..." 的行
            match = re.match(r'^(\d+):\s+([^|]+)\s*\|(.*)', line.strip())
            if match:
                step = int(match.group(1))
                total_loss = float(match.group(2).strip())
                metrics_str = match.group(3)
                
                # 解析各个指标
                metrics = {'step': step, 'total_loss': total_loss}
                
                # 提取各个loss
                for metric_match in re.finditer(r'(\w+_loss):\s*([^|]+)', metrics_str):
                    metric_name = metric_match.group(1)
                    metric_value = float(metric_match.group(2).strip())
                    metrics[metric_name] = metric_value
                
                metrics_data.append(metrics)
    
    return metrics_data

# 从训练日志中提取数据
root_dir = "./logs/dc_smac/3m-Good/" + "hh_10-bcw_1.0-qw_1.0-pw_0.1-guidew_2.0-lr_1e-05-n_ddim_2-10021/10021"
log_file = f"{root_dir}/outputs.log"
metrics_data = extract_metrics_from_log(log_file)

print(f"提取到 {len(metrics_data)} 个训练步骤的数据")
if metrics_data:
    print("\n前5个步骤的数据:")
    for i, data in enumerate(metrics_data[:5]):
        print(f"Step {data['step']}: {data}")
    
    print(f"\n最后5个步骤的数据:")
    for i, data in enumerate(metrics_data[-5:]):
        print(f"Step {data['step']}: {data}")
    
    # 保存为DataFrame便于分析
    df = pd.DataFrame(metrics_data)
    print(f"\n数据统计:")
    print(df.describe())
    
    # 保存为CSV
    df.to_csv(f'{root_dir}/extracted_metrics.csv', index=False)
    print("\n数据已保存到 extracted_metrics.csv")
else:
    print("没有找到有效的metrics数据")
