from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot

import numpy as np

gasf_transformer = GramianAngularField(method="summation")
gadf_transformer = GramianAngularField(method="difference")
mtf_transformer = MarkovTransitionField(n_bins=10, strategy="normal")
rp_transformer = RecurrencePlot()

def gasf_transform(segments):
    segment_array = []
    for segment in segments:
        segment_array.append(
            gadf_transformer.fit_transform(segment.reshape(1, -1))[0]
        )
    return np.asarray(segment_array)

def gadf_transform(segments):
    segment_array = []
    for segment in segments:
        segment_array.append(
            gasf_transformer.fit_transform(segment.reshape(1, -1))[0]
        )
    return np.asarray(segment_array)

def mtf_transform(segments):
    segment_array = []
    for segment in segments:
        segment_array.append(
            mtf_transformer.fit_transform(segment.reshape(1, -1))[0]
        )
    return np.asarray(segment_array)

def rp_transform(segments):
    segment_array = []
    mean = np.mean(segments)
    for segment in segments:
        segment_array.append(
            rp_transformer.fit_transform(segment.reshape(1, -1))[0]
        )
    return np.asarray(segment_array) + mean