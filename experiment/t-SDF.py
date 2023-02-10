from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
import numpy as np
import glob as glob
import sys

before_fine_feature_path_list = glob.glob("experiment/test_features_wo_finetuning/*")
after_fine_feature_path_list = glob.glob("experiment/test_features_w_finetuning/*")


def T_SNE(feature_path_list):
    feature_list = np.zeros((1,8192))
    for i, data in enumerate(feature_path_list):
        feature_list = np.vstack((feature_list, np.load(data).reshape(1, 8192)))
    print("load complete!")

    # 2차원으로 차원 축소
    n_components = 2

    # t-sne 모델 생성
    model = TSNE(n_components=n_components, verbose=True)

    result = np.array(model.fit_transform(feature_list))

    return result


no_fine_result = T_SNE(before_fine_feature_path_list)
fine_result = T_SNE(after_fine_feature_path_list)

plt.plot(fine_result[:, 0], fine_result[:, 1], '.b')
plt.plot(no_fine_result[:, 0], no_fine_result[:, 1], '.r')

plt.show()
