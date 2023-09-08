import os
import json
import csv

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

class NdarrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NdarrayEncoder, self).default(obj)


class IOUtils:
    @staticmethod
    def write_to_file(message, file_name):
        with open(file_name, 'a') as f:
            if isinstance(message, dict):
                json.dump(message, f, indent=4, cls=NdarrayEncoder)
            else:
                f.write(message)
            f.write('\n')

    @staticmethod
    def list_to_csv(data, file_name):
        with open(file_name, 'w') as file:
            writer = csv.writer(file, lineterminator='\n')
            writer.writerows(data)

    @staticmethod
    def load_from_json(file):
        with open(file, 'r') as f:
            dict = json.load(f)
        return dict


def save_history_loss_and_acc(history, save_dir, figsize=(13, 9)):
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    fig.subplots_adjust(hspace=0.5)
    axes[0].plot(history.history['loss'])
    axes[0].plot(history.history['val_loss'])
    axes[0].set_title('Loss')
    axes[0].set_ylabel('loss')
    axes[0].set_xlabel('epoch')
    axes[0].legend(['train', 'test'], loc='best')

    axes[1].plot(history.history['accuracy'])
    axes[1].plot(history.history['val_accuracy'])
    axes[1].set_title('Accuracy')
    axes[1].set_ylabel('accuracy')
    axes[1].set_xlabel('epoch')
    axes[1].legend(['train', 'valid'], loc='best')

    fig.savefig(os.path.join(save_dir, 'loss_and_acc.jpg'))


class ConfusionMatrixUtils:
    @staticmethod
    def normarized_confusion_matrix(cm_num):
        """混合行列（正規化）を返す
        Args:
            cm_num(list): 混合行列（数）
        Returns:
            cm_normarized(list): 混合行列（正規化）
        """
        # 0-1に正規化する
        cm_num = cm_num.astype(np.float32)
        cm_normarized = cm_num / cm_num.sum(axis=1, keepdims=True)
        return cm_normarized

    @staticmethod
    def save_normarized_confusion_matrix_img(cm_normarized, save_dir, lebel_name_list):
        """混合行列をきれいに図示"""
        fig, ax = plt.subplots(figsize=(12, 10))  # figsize=(12, 10)
        ax.set_title('Normalized confusion matrix')
        ax.set_ylabel('Predicted label')
        ax.set_xlabel('True label')
        sns.heatmap(
            cm_normarized,
            ax=ax,
            annot=True,
            cmap='Blues',
            xticklabels=lebel_name_list,
            yticklabels=lebel_name_list,
            vmax=1,
            vmin=0)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir / 'normalized_confusion_matrix.png')


def main():
    return


if __name__ == '__main__':
    main()