import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from scipy import mean



# 指定日志文件的路径
file_path = '/data_1/wxl22/MIOOD/logs/sota_label_ec50_core-2.0-1-20231229-231017/auc.txt'
file_path_full = "/data_1/wxl22/MIOOD/logs/sota_label_ec50_core-2.0-1-20231229-231017/auc_all.txt"
file_path_nodo = "/data_1/wxl22/MIOOD/logs/label_ec50_core-2.0-0.0-20240105-160737/auc_all.txt"
file_path_nocl = "/data_1/wxl22/MIOOD/logs/label_ec50_core-0.0-1.0-20240103-205922/auc_all.txt"
file_path_noboth = "/data_1/wxl22/MIOOD/logs/label_ec50_core-0.0-0.0-20240104-143608/auc_all.txt"
file_path_nofusion = "/data_1/wxl22/MIOOD_nofusion/logs/label_ec50_core-2.0-1.0-20240112-211940/auc_all.txt"

def extract_auc_from_logs(file_path):
    # 从文件中读取日志文本
    with open(file_path, 'r') as file:
        log_text = file.read()

    # 使用正则表达式查找所有valid部分的auc值
    auc_values = re.findall(r"valid: \{'auc': ([0-9.]+),", log_text)

    # 将找到的字符串auc值转换为浮点数
    auc_values = [float(auc) for auc in auc_values]

    return auc_values

def confidence_interval(data):
    confidence = 0.95
    n = len(data)
    m = np.mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return m - h, m + h

# 读取数据
auc_values_full = extract_auc_from_logs(file_path_full)
auc_values_nocl = extract_auc_from_logs(file_path_nocl)
auc_values_nodo = extract_auc_from_logs(file_path_nodo)
auc_values_noboth = extract_auc_from_logs(file_path_noboth)
auc_values_nofusion = extract_auc_from_logs(file_path_nofusion)
# 将每个数组分为四个50维的子数组
split_arrays1 = np.split(np.array(auc_values_full), 4)
split_arrays2 = np.split(np.array(auc_values_nocl), 4)
split_arrays3 = np.split(np.array(auc_values_nodo), 4)
split_arrays4 = np.split(np.array(auc_values_noboth), 4)
split_arrays5 = np.split(np.array(auc_values_nofusion), 4)
# 绘制置信区间图
x = np.arange(0, 50)
plt.figure(figsize=(10, 6))

nocl = [174/255, 78/255, 137/255]
# full = [166/255, 32/255, 46/255]
nodo = [234/255, 141/255, 158/255]
full = [59/255, 144/255, 217/255]
noboth = [230/255, 185/255, 117/255]
nofusion = [94/255, 177/255, 172/255]
colors = [nocl, nodo, noboth, nofusion, full]  # 不同的颜色
markers = ['o', '^', 's', 'd', 'x']  # 不同的标记
labels = ['w/o IFE ', 'w/o DI', 'w/o IFE+DI', 'w/o HMR', 'Full']
# 为每个数组绘制折线图和置信区间
for split_array, label, color, marker in zip([split_arrays2, split_arrays3, split_arrays4, split_arrays5, split_arrays1],labels, colors, markers):
    mean_values = np.mean(split_array, axis=0)
    ci_lower, ci_upper = zip(*[confidence_interval(data) for data in np.array(split_array).T])
    plt.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.15)
    plt.plot(x, mean_values, label=label, color=color, marker=marker, markersize=4.5)

# plt.title('AUC with Confidence Intervals', fontsize=18)
plt.xlabel('Epoch',fontsize=16)
plt.ylabel('AUC', fontsize=16)
plt.legend(fontsize=12, ncol=5, loc='upper center')
plt.xlim([0, 50])  # 设置 x 轴的范围从 0 开始
plt.ylim([0.2, 0.9])  # 根据数据实际情况调整y轴范围
plt.savefig('/data_1/wxl22/MIOOD/utils/5condition_combined_line_chart_with_ci.png', dpi=900)
plt.show()