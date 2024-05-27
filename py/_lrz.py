# 23-12-29

import numpy as np
from _errlog import *


def _lrz1d(data: np.ndarray, pred_err: np.ndarray):
    if not len(data.shape) == 1:
        raise ValueError("dimension mismatch: " + f"current size is {data.shape}")
    pred_err[1:] = data[1:] - data[:-1]
    pred_err[0] = data[0]


def _lrz1d_decomp(pred_err: np.ndarray, xdata: np.ndarray):
    if not len(xdata.shape) == 1:
        raise ValueError("dimension mismatch: " + f"current size is {xdata.shape}")
    xdata[:] = np.cumsum(pred_err)[:]


def _lrz2d(data: np.ndarray, pred_err: np.ndarray):
    if not len(data.shape) == 2:
        raise ValueError("dimension mismatch: " + f"current size is {data.shape}")
    pred_err[1:, 1:] = data[1:, 1:] - (data[1:, :-1] + data[:-1, 1:] - data[:-1, :-1])
    pred_err[0, 0] = data[0, 0]
    pred_err[0, 1:] = data[0, 1:] - data[0, :-1]
    pred_err[1:, 0] = data[1:, 0] - data[:-1, 0]
    # print(extrema(pred_err))


def _lrz2d_decomp(pred_err: np.ndarray, xdata: np.ndarray):
    xdata[:, :] = np.cumsum(pred_err, axis=0)
    xdata[:, :] = np.cumsum(xdata, axis=1)


def _lrz3d(data: np.ndarray, pred_err: np.ndarray):
    if not len(data.shape) == 3:
        raise ValueError("dimension mismatch: " + f"current size is {data.shape}")

    padded_shape = np.array(data.shape) + np.array([1, 1, 1])
    space = np.zeros(padded_shape, dtype=data.dtype)
    space[1:, 1:, 1:] = data[:, :, :]
    pred_err[:, :, :] = space[1:, 1:, 1:] - (
        space[1:, 1:, :-1]
        + space[1:, :-1, 1:]
        + space[:-1, 1:, 1:]
        - space[1:, :-1, :-1]
        - space[:-1, 1:, :-1]
        - space[:-1, :-1, 1:]
        + space[:-1, :-1, :-1]
    )


def _lrz3d_decomp(pred_err: np.ndarray, xdata: np.ndarray):
    xdata[:, :, :] = np.cumsum(pred_err, axis=0)
    xdata[:, :, :] = np.cumsum(xdata, axis=1)
    xdata[:, :, :] = np.cumsum(xdata, axis=2)


class Sz14Approx:
    @staticmethod
    def lorenzo_predict(data: np.ndarray, pred_err: np.ndarray):
        if len(data.shape) == 1:
            # print("lrz1d")
            _lrz1d(data, pred_err)
        elif len(data.shape) == 2:
            # print("lrz2d")
            _lrz2d(data, pred_err)
        elif len(data.shape) == 3:
            # print("lrz3d")
            _lrz3d(data, pred_err)
        else:
            raise ValueError("data.shape other than {1,2,3} is not supported")
        # print(extrema(pred_err))

    @staticmethod
    def quantize(
        d: np.ndarray, d_pre: np.ndarray, eb, in_place=False, cutoff_range=None
    ):
        """
        with unlimited value range for prediction error, quantized
        """

        if cutoff_range:
            raise NotImplementedError("cutoff-range functionality is not implemented")

        ebx2_r = 1 / (2 * eb)
        if in_place:
            if d_pre is None:
                d.ravel()[:] = np.round(d.ravel()[:] * ebx2_r, 0)
            else:
                raise ValueError("2nd arg should be None when in_place=True")
        else:
            if d_pre is None:
                raise ValueError("2nd arg CANNOT be None when in_place=False")
            d_pre.ravel()[:] = np.round(d.ravel()[:] * ebx2_r, 0)


class PszApprox:
    @staticmethod
    def preceding_quantize(data: np.ndarray, data_quantized: np.ndarray, eb):
        ebx2_r = 1 / (2 * eb)
        data_quantized.ravel()[:] = np.round(data.ravel()[:] * ebx2_r, 0)

    @staticmethod
    def reverse_quantize(xdata_quantized: np.ndarray, xdata: np.ndarray, eb):
        xdata.ravel()[:] = np.round(xdata_quantized.ravel()[:] * (eb * 2), 0)

    @staticmethod
    def succeeding_lorenzo_predict(data_quntized: np.ndarray, pred_err: np.ndarray):
        if len(data_quntized.shape) == 1:
            _lrz1d(data_quntized, pred_err)
        elif len(data_quntized.shape) == 2:
            _lrz2d(data_quntized, pred_err)
        elif len(data_quntized.shape) == 3:
            _lrz3d(data_quntized, pred_err)
        else:
            raise ValueError("data_quntized.shape other than {1,2,3} is not supported")

    @staticmethod
    def reverse_lorenzo_predict(pred_err: np.ndarray, xdata_quntized: np.ndarray):
        if len(xdata_quntized.shape) == 1:
            _lrz1d_decomp(pred_err, xdata_quntized)
        elif len(xdata_quntized.shape) == 2:
            _lrz2d_decomp(pred_err, xdata_quntized)
        elif len(xdata_quntized.shape) == 3:
            _lrz3d_decomp(pred_err, xdata_quntized)
        else:
            raise ValueError("xdata_quntized.shape other than {1,2,3} is not supported")
