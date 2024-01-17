
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np






def draw_auc_accuracy(eva_method, data_path):

    with open(f'{data_path}/draw_data_2019.json', 'r') as f:
        data = json.load(f)
        valid_ood = data['valid_ood']
    valid_ood_auc = [loss for loss in valid_ood]

    # 绘制折线图，使用不同颜色
    plt.plot(valid_ood_auc, color='red', label=f'valid_ood_{eva_method}', linewidth=0.5)


    # 添加标题和标签
    plt.title(f'valid {eva_method}')



    plt.xticks(list(range(0, len(valid_ood_auc), 20)))
    plt.xlabel('Epoch')
    plt.ylabel(f'{eva_method}')

    # 添加图例
    plt.legend()

    # 显示图形
    # plt.show()
    plt.savefig(f'{data_path}/image: accuracy_curve.png')

if __name__ == '__main__':
    pass