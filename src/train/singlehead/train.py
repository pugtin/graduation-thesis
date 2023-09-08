from model_structure import create_model
from settings import get_wisdm_settings, get_wheelchair_settings
import task_utils
import utils

import sys
import yaml
import numpy as np
import h5py
import collections

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE, RandomOverSampler

def load_data():
    X_dict = {}
    y_dict = {}

    for subject in subjects:
        for subject in subjects:
            h5path = processed_path / str(subject) / f"{encoding}.h5"
            h5f = h5py.File(h5path, "r")
            shape = h5f["shape"][:]
            single = np.array(h5f[f"{encoding}_single"][:]).reshape(shape)
            h5f.close()
            # single_path = processed_path / str(subject) / encoding / "single.npy"
            #
            # single = np.load(single_path).astype("float32")
            X_dict[subject] = single

            y = np.load(label_path / f"{subject}.npy")
            y_dict[subject] = y

        return X_dict, y_dict

def count_label(one_hot_label, label):
    d = collections.defaultdict(int)
    for k, v in collections.Counter(np.argmax(one_hot_label, axis=1)).items():
        d[label[int(k)]] += v
    return d


def train_and_predict(X_train, X_val, X_test, y_train_hot, y_val_hot, save_dir, model):
    callbacks = []
    # if settings.SAVE_MODEL:
    #     model_filename = os.path.join(save_dir, 'model.h5')
    #     callbacks.append(ModelCheckpoint(filepath=model_filename, monitor = 'val_loss', save_weights_only=True, save_best_only=True))
    callbacks.append(EarlyStopping(monitor='val_loss', patience=patience))#, restore_best_weights=True))
    sample_weights=class_weight.compute_sample_weight("balanced", y_train_hot)
    history = model.fit(
        X_train,
        y_train_hot,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        sample_weight=sample_weights,
        callbacks=callbacks,
        validation_data=(X_val, y_val_hot),
    )

    pred_test_arr = model.predict(X_test)
    pred_test_list = np.argmax(pred_test_arr, axis=1)

    return history, model, pred_test_list


def save_outputs(history, save_dir, y_test, pred_test_list):
    utils.save_history_loss_and_acc(history, save_dir)
    task_utils.save_label_predict_as_json(
        y_test,
        pred_test_list,
        save_dir / "label_predict.json"
    )

    score_dict = classification_report(y_test, pred_test_list, output_dict=True)
    score_dict['accuracy'] = np.mean(np.array(y_test) == np.array(pred_test_list))
    utils.IOUtils.write_to_file(score_dict, f'{save_dir}/score_dict.json')
    task_utils.ConfusionMatrix.save_confusion_matrix_img(save_dir, label_rule=labels)

def train_loso():
    X_dict, y_dict = load_data()

    for test_subject in subjects:
        print(f'---------- user {test_subject} ----------')

        # Making the copy
        train_subjects = subjects[:]
        train_subjects.remove(test_subject)
        print(train_subjects)
        X_train = np.vstack([X_dict[subject] for subject in train_subjects])
        y_train = np.hstack([y_dict[subject] for subject in train_subjects])

        #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)
        rus = SMOTE()
        if len(set(y_train)) > 1:
            X_rus, y_rus = rus.fit_resample(X_train.reshape(y_train.shape[0], -1), y_train)
            X_train = X_rus.reshape(y_rus.shape[0], width, height, channels)
            y_train = y_rus

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

        X_test = X_dict[test_subject]
        y_test = y_dict[test_subject]

        y_train_hot = to_categorical(y_train, len(labels))
        y_val_hot = to_categorical(y_val, len(labels))
        y_test_hot = to_categorical(y_test, len(labels))
        print(X_train.shape, X_val.shape, X_test.shape)
        print(count_label(y_train_hot, y_train))
        save_dir = log_dir / str(test_subject)
        save_dir.mkdir(exist_ok=True)
        model = create_model(input_shape=(width, height, channels), output_dim=y_test_hot.shape[1])
        history, model, pred_test_list = train_and_predict(X_train, X_val, X_test, y_train_hot, y_val_hot, save_dir, model)
        save_outputs(history, save_dir, np.argmax(y_test_hot, axis=1), pred_test_list)

    task_utils.ConfusionMatrix.save_confusion_matrix_img_all(log_dir, subjects=subjects, label_rule=labels)
    task_utils.save_avg_f1_and_acc_all(log_dir, subjects=subjects)


if __name__ == '__main__':
    # Command Rule: python3 train_loso.py WISDM gasf
    dataset = sys.argv[1].lower()
    encoding = sys.argv[2].lower()

    if dataset == "wisdm":
        setting = get_wisdm_settings()
    elif dataset == "wheelchair":
        setting = get_wheelchair_settings()
    else:
        print("Invalid arugment given")
        sys.exit(-1)

    yaml_file = open(setting.yaml_path, "r")
    yaml_config = yaml.safe_load(yaml_file)

    processed_path = setting.processed_path
    label_path = setting.label_path
    if encoding == "mtf":
        log_dir = setting.mtf_dir
    elif encoding == "rp":
        log_dir = setting.rp_dir
    elif encoding == "gasf":
        log_dir = setting.gasf_dir
    elif encoding == "gadf":
        log_dir = setting.gadf_dir
    else:
        print("Invalid arugment given")
        sys.exit(-1)

    log_dir.mkdir(exist_ok=False)

    subjects = yaml_config[0]["subjects"]
    labels = yaml_config[1]["labels"]

    width = setting.width
    height = setting.height
    channels = setting.channels
    patience = setting.patience
    epochs = setting.epochs
    batch_size = setting.batch_size

    train_loso()