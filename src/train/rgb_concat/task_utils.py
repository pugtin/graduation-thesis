import utils

import json
import collections

import numpy as np
from sklearn.metrics import confusion_matrix

def save_label_predict_as_json(label_list, predict_list, save_file_name):
    label_predict = {
        'label': label_list,
        'predict': predict_list
    }
    utils.IOUtils.write_to_file(label_predict, save_file_name)


def save_avg_f1_and_acc_all(dir, subjects):
    f1_total = 0
    acc_total = 0
    with open(f'{dir}/score.txt', 'w') as f:
        for subject in subjects:
            f.write(f'------{subject}------\n')
            file = f'{dir}/{subject}/score_dict.json'
            dict = utils.IOUtils.load_from_json(file)
            f1 = dict['macro avg']['f1-score']
            acc = dict['accuracy']
            f1_total += f1
            acc_total += acc
            f.write(f'f1: {f1}\n')
            f.write(f'acc: {acc}\n')
        f1_avg = f1_total / len(subjects)
        acc_avg = acc_total / len(subjects)
        f.write('-'*20+'\n')
        f.write(f'f1_avg: {f1_avg}\n')
        f.write(f'acc_avg: {acc_avg}\n')


class ConfusionMatrix:
    @staticmethod
    def _load_predict_label(save_dir):
        file = f'{save_dir}/label_predict.json'
        with open(file,'r') as f:
            dict = json.load(f)
        predict_list = dict['predict']
        label_list = dict['label']
        return label_list, predict_list

    @staticmethod
    def _create_and_save_confusion_matrix_img(label_list_all, predict_list_all, lebel_list, save_dir, label_rule):
        def _label2name(label_num_list, label_rule):
            return [label2name(label_num, label_rule) for label_num in label_num_list]

        cm_num = confusion_matrix(label_list_all, predict_list_all, labels=lebel_list)
        cm_normarized = utils.ConfusionMatrixUtils.normarized_confusion_matrix(cm_num)
        lebel_name_list = _label2name(lebel_list, label_rule)
        utils.ConfusionMatrixUtils.save_normarized_confusion_matrix_img(cm_normarized, save_dir, lebel_name_list)

    @staticmethod
    def save_confusion_matrix_img(save_dir, label_rule):
        def _create_label_list(save_dir):
            label_list, predict_list = ConfusionMatrix._load_predict_label(save_dir)
            lebel_set = set(label_list) | set(predict_list)
            lebel_list = sorted(list(lebel_set))
            return label_list, predict_list, lebel_list

        label_list, predict_list, lebel_list = _create_label_list(save_dir)
        ConfusionMatrix._create_and_save_confusion_matrix_img(label_list, predict_list, lebel_list, save_dir, label_rule)

    @staticmethod
    def save_confusion_matrix_img_all(save_dir, subjects, label_rule):
        def _create_label_list_all(save_dir):
            lebel_set = set()
            label_list_all = []
            predict_list_all = []
            for test_subject in subjects:
                fold_dir = f'{save_dir}/{test_subject}'

                label_list, predict_list = ConfusionMatrix._load_predict_label(fold_dir)
                label_list_all.extend(label_list)
                predict_list_all.extend(predict_list)
                lebel_set |= set(label_list) | set(predict_list)
            lebel_list = sorted(list(lebel_set))
            return label_list_all, predict_list_all, lebel_list

        label_list_all, predict_list_all, lebel_list = _create_label_list_all(save_dir)
        ConfusionMatrix._create_and_save_confusion_matrix_img(label_list_all, predict_list_all, lebel_list, save_dir, label_rule)

def label2name(label_num, label_rule):
    for key, value in label_rule.items():
        if value == label_num:
            return key
    return None

def main():
    return


if __name__ == '__main__':
    main()