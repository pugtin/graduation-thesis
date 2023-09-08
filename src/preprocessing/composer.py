from settings import get_wisdm_settings, get_wheelchair_settings
from utils import create_dir, one_to_one_scaler, label_converter, rgb_lookup_table_converter, rgb_scaler
from encoder import gasf_transform, gadf_transform, mtf_transform, rp_transform

from skimage.transform import resize
import sys
import numpy as np
import yaml
import h5py

# Load Yaml
wisdm_yaml = open(get_wisdm_settings().yaml_path, "r")
wisdm_yaml = yaml.safe_load(wisdm_yaml)

wheelchair_yaml = open(get_wheelchair_settings().yaml_path, "r")
wheelchair_yaml = yaml.safe_load(wheelchair_yaml)

def advanced_converter(mode):
    for processed_path in list(processed_paths.iterdir()):
        if processed_path.name.startswith("."):
            continue
        gasf = [np.load(processed_path / "gasf" / "x.npy"),
                np.load(processed_path / "gasf" / "y.npy"),
                np.load(processed_path / "gasf" / "z.npy")
                ]
        gadf = [np.load(processed_path / "gadf" / "x.npy"),
                np.load(processed_path / "gadf" / "y.npy"),
                np.load(processed_path / "gadf" / "z.npy")
                ]
        mtf = [np.load(processed_path / "mtf" / "x.npy"),
                np.load(processed_path / "mtf" / "y.npy"),
                np.load(processed_path / "mtf" / "z.npy")
               ]
        rp = [np.load(processed_path / "rp" / "x.npy"),
                np.load(processed_path / "rp" / "y.npy"),
                np.load(processed_path / "rp" / "z.npy")
              ]

        # gasf + mtf + rp
        if mode == "fusion1":
            create_dir(processed_path / "fusion1")
            x = np.stack(list([gasf[0], mtf[0], rp[0]]), axis=1)
            y = np.stack(list([gasf[1], mtf[1], rp[1]]), axis=1)
            z = np.stack(list([gasf[2], mtf[2], rp[2]]), axis=1)

            h5f = h5py.File(processed_path / "fusion1.h5", "w")
            h5f.create_dataset("shape", x.shape)
            h5f.create_dataset("fusion1_x", x.ravel())
            h5f.create_dataset("fusion1_y", y.ravel())
            h5f.create_dataset("fusion1_z", z.ravel())
            h5f.close()

            # np.save(processed_path / "fusion1" / "x.npy", x)
            # np.save(processed_path / "fusion1" / "y.npy", y)
            # np.save(processed_path / "fusion1" / "z.npy", z)

        # gadf + mtf + rp
        elif mode == "fusion2":
            create_dir(processed_path / "fusion2")
            x = np.stack(list([gadf[0], mtf[0], rp[0]]), axis=1)
            y = np.stack(list([gadf[1], mtf[1], rp[1]]), axis=1)
            z = np.stack(list([gadf[2], mtf[2], rp[2]]), axis=1)

            h5f = h5py.File(processed_path / "fusion2.h5", "w")
            h5f.create_dataset("shape", x.shape)
            h5f.create_dataset("fusion2_x", x.ravel())
            h5f.create_dataset("fusion2_y", y.ravel())
            h5f.create_dataset("fusion2_z", z.ravel())
            h5f.close()

            # np.save(processed_path / "fusion2" / "x.npy", x)
            # np.save(processed_path / "fusion2" / "y.npy", y)
            # np.save(processed_path / "fusion2" / "z.npy", z)

        elif mode == "rgb_concat":
            create_dir(processed_path / "rgb_concat")
            h5f = h5py.File(processed_path / "rgb_concat.h5", "w")
            reds, greens, blues = [], [], []
            for gasf_acc in gasf[:3]:
                r, g, b = \
                resize(
                    rgb_lookup_table_converter(
                        rgb_scaler(gasf_acc)
                    )[:,:,:,0], (128, 128)
                ),\
                resize(
                    rgb_lookup_table_converter(
                        rgb_scaler(gasf_acc)
                    )[:,:,:,1], (128, 128)
                ),\
                resize(
                    rgb_lookup_table_converter(
                        rgb_scaler(gasf_acc)
                    )[:,:,:,2], (128, 128)
                )

                reds.append(r)
                greens.append(g)
                blues.append(b)

            # np.save(processed_path / "rgb_concat" / "gasf.npy",
            #         one_to_one_scaler(
            #             np.stack(
            #                 [
            #                     np.dstack(reds), np.dstack(greens), np.dstack(blues)
            #                  ], axis=1)
            #         )
            # )
            h5f.create_dataset("shape", np.stack(
                [np.dstack(reds), np.dstack(greens), np.dstack(blues)]
                , axis=1).shape)
            h5f.create_dataset("rgb_concat_gasf", one_to_one_scaler(
                np.stack(
                    [
                        np.dstack(reds), np.dstack(greens), np.dstack(blues)
                     ], axis=1)
            ))

            reds, greens, blues = [], [], []
            for gadf_acc in gadf[:3]:
                r, g, b = \
                resize(
                    rgb_lookup_table_converter(
                        rgb_scaler(gadf_acc)
                    )[:,:,:,0], (128, 128)
                ),\
                resize(
                    rgb_lookup_table_converter(
                        rgb_scaler(gadf_acc)
                    )[:,:,:,1], (128, 128)
                ),\
                resize(
                    rgb_lookup_table_converter(
                        rgb_scaler(gadf_acc)
                    )[:,:,:,2], (128, 128)
                )

                reds.append(r)
                greens.append(g)
                blues.append(b)

            h5f.create_dataset("rgb_concat_gadf", one_to_one_scaler(
                np.stack(
                    [
                        np.dstack(reds), np.dstack(greens), np.dstack(blues)
                     ], axis=1)
            ))

            reds, greens, blues = [], [], []
            for mtf_acc in mtf[:3]:
                r, g, b = \
                resize(
                    rgb_lookup_table_converter(
                        rgb_scaler(mtf_acc)
                    )[:,:,:,0], (128, 128)
                ),\
                resize(
                    rgb_lookup_table_converter(
                        rgb_scaler(mtf_acc)
                    )[:,:,:,1], (128, 128)
                ),\
                resize(
                    rgb_lookup_table_converter(
                        rgb_scaler(mtf_acc)
                    )[:,:,:,2], (128, 128)
                )

                reds.append(r)
                greens.append(g)
                blues.append(b)

            h5f.create_dataset("rgb_concat_mtf", one_to_one_scaler(
                np.stack(
                    [
                        np.dstack(reds), np.dstack(greens), np.dstack(blues)
                     ], axis=1)
            ))

            h5f.close()

def base_converter(mode):
    for segment_path, label_path in list(zip(list(segment_paths.iterdir()), list(label_paths.iterdir()))):
        create_dir(processed_paths / segment_path.name.split(".")[0])
        _current_dir = processed_paths / label_path.name.split(".")[0]

        label_lookup = dict((v, k) for k, v in yaml_config[1]["labels"].items())

        segments = np.load(segment_path)

        if mode == "gasf":
            # Rescale the segment data per user from -1, 1 before
            segments = one_to_one_scaler(segments)
            x_encode, y_encode, z_encode = [gasf_transform(segments[:,:,n]) for n in range(segments.shape[-1])]
            single = np.stack([x_encode, y_encode, z_encode], axis=1)

            label = np.asarray(label_converter(label_lookup, np.load(label_path)))
            # GASF
            # X = rgb_scaler(gasf(np.load(segment)))
            # label = np.asarray(label_converter(label_lookup, np.load(label)))
        elif mode == "gadf":
            # Rescale the segment data per user from -1, 1 before
            segments = one_to_one_scaler(segments)
            x_encode, y_encode, z_encode = [gadf_transform(segments[:,:,n]) for n in range(segments.shape[-1])]
            single = np.stack([x_encode, y_encode, z_encode], axis=1)

            label = np.asarray(label_converter(label_lookup, np.load(label_path)))

        elif mode == "mtf":
            # In order to follow the Gaussian distribution for the number of bins, strategy is normal during
            x_encode, y_encode, z_encode = [mtf_transform(segments[:,:,n]) for n in range(segments.shape[-1])]
            single = np.stack([x_encode, y_encode, z_encode], axis=1)

            label = np.asarray(label_converter(label_lookup, np.load(label_path)))
        elif mode == "rp":
            # Calculate the mean per activities of each users and add the values after
            x_encode, y_encode, z_encode = [rp_transform(segments[:, :, n]) for n in range(segments.shape[-1])]
            single = np.stack([x_encode, y_encode, z_encode], axis=1)

            label = np.asarray(label_converter(label_lookup, np.load(label_path)))

        x_encode, y_encode, z_encode, single = one_to_one_scaler(x_encode),\
                                               one_to_one_scaler(y_encode),\
                                               one_to_one_scaler(z_encode),\
                                               one_to_one_scaler(single)

        h5f = h5py.File(_current_dir / f"{mode}.h5", "w")
        h5f.create_dataset("shape", x_encode.shape)
        h5f.create_dataset(f"{mode}_x", x_encode)
        h5f.create_dataset(f"{mode}_y", y_encode)
        h5f.create_dataset(f"{mode}_z", z_encode)
        h5f.create_dataset(f"{mode}_single", single)

        h5f.close()

        # np.save(_mode_dir / "x.npy", x_encode)
        # np.save(_mode_dir / "y.npy", y_encode)
        # np.save(_mode_dir / "z.npy", z_encode)
        # np.save(_mode_dir / "single.npy", single)

        # Processing as image
        # for num, x in enumerate(X):
        #     image_path = _mode_dir / f"{num}_{label[num]}.png"
        #     save_image(x.T, image_path, channel=channel)

if __name__ == "__main__":
    dataset = sys.argv[1]
    mode = sys.argv[2]

    if dataset == "wheelchair":
        segment_paths = get_wheelchair_settings().segment_path
        label_paths = get_wheelchair_settings().label_path
        processed_paths = get_wheelchair_settings().processed_path

        yaml_config = wheelchair_yaml

    elif dataset == "wisdm":
        segment_paths = get_wisdm_settings().segment_path
        label_paths = get_wisdm_settings().label_path
        processed_paths = get_wisdm_settings().processed_path

        yaml_config = wisdm_yaml
    if mode in ["gadf", "gasf", "rp", "mtf"]:
        base_converter(mode)
    else:
        advanced_converter(mode)

