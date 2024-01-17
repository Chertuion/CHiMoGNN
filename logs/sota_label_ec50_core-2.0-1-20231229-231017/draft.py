import re
import numpy as np
import matplotlib.pyplot as plt
def extract_auc_from_logs(file_path):
    # 从文件中读取日志文本
    with open(file_path, 'r') as file:
        log_text = file.read()

    # 使用正则表达式查找所有valid部分的auc值
    auc_values = re.findall(r"valid: \{'auc': ([0-9.]+),", log_text)
    
    # 将找到的字符串auc值转换为浮点数
    auc_values = [float(auc) for auc in auc_values]

    return auc_values

# 指定日志文件的路径
file_path = '/data_1/wxl22/MIOOD/logs/sota_label_ec50_core-2.0-1-20231229-231017/auc.txt'

# 调用函数并打印结果
auc_values_full = extract_auc_from_logs("/data_1/wxl22/MIOOD/logs/sota_label_ec50_core-2.0-1-20231229-231017/auc_all.txt")
auc_values_nocl = extract_auc_from_logs("/data_1/wxl22/MIOOD/logs/label_ec50_core-2.0-0.0-20240105-160737/auc_all.txt")
auc_values_nodo = extract_auc_from_logs("/data_1/wxl22/MIOOD/logs/label_ec50_core-0.0-1.0-20240103-205922/auc_all.txt")
auc_values_noboth = extract_auc_from_logs("/data_1/wxl22/MIOOD/logs/label_ec50_core-0.0-0.0-20240104-143608/auc_all.txt")



# 计算四个数组在对应位置的平均值
auc_values_full = np.mean(np.split(np.array(auc_values_full), 4), axis=0)
auc_values_nocl = np.mean(np.split(np.array(auc_values_nocl), 4), axis=0)
auc_values_nodo = np.mean(np.split(np.array(auc_values_nodo), 4), axis=0)
auc_values_noboth = np.mean(np.split(np.array(auc_values_noboth), 4), axis=0)

print("nocl: ", len(auc_values_nocl), max(auc_values_nocl))
print("nodo: ", len(auc_values_nodo), max(auc_values_nodo))
print("noboth: ", len(auc_values_noboth), max(auc_values_noboth))
print("Full: ", len(auc_values_full), max(auc_values_full))

x = np.arange(0, 50)
# 绘制折线图
plt.plot(x, auc_values_noboth, label='no both')
plt.plot(x, auc_values_nodo, label='no do')
plt.plot(x, auc_values_nocl, label='no cl')
plt.plot(x, auc_values_full, label='full')

# 添加标题和轴标签
plt.title('Four Lines on One Graph')
plt.xlabel('X axis')
plt.ylabel('Y axis')
# 设置y轴范围
plt.ylim([0, 1])
# 显示图例
plt.legend()
# 保存图表为PNG文件，分辨率为900 DPI
plt.savefig('/data_1/wxl22/MIOOD/utils/line_chart.png', dpi=900)
# 显示图表
plt.show()