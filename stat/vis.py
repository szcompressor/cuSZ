__author__ = "Jiannan Tian"
__copyright__ = "(C) 2021 by Washington State University, Argonne National Laboratory"
__license__ = "BSD 3-Clause"
__version__ = "0.2.x"
__date__ = "2021-03-29"
__change__ = "(rev) 2021-03-30"

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import matplotlib.pyplot as plt
from smoothness import *


def do_linear_regression(input_x, input_y):
    regr = linear_model.LinearRegression()
    regr.fit(input_x[:, np.newaxis], input_y)
    pred_y = regr.predict(input_x[:, np.newaxis])
    return pred_y, regr.intercept_, regr.coef_, mean_squared_error(input_y, pred_y), r2_score(input_y, pred_y)


def helper_fmt_label(intercept, coef, r2) -> str:
    return f"intercept: {intercept:.3f}, coef: {coef[0]:.3e}, r2 :{r2:.3e}"


def process_plot_3fig(ana_ori: AnalyzeSmoothnessWholeDatum,
                      ana_qua: AnalyzeSmoothnessWholeDatum,
                      end_at: int, folder_name: str, name: str,
                      plot=True,
                      show_plot=False
                      ):
    xaxis_range = np.arange(len(ana_ori.var_abs_1d))
    if end_at > len(xaxis_range):
        print("ending too large, adjusting...")
        end_at = len(xaxis_range)

    if not plot:
        return None

    fig, _ax = plt.subplots(nrows=3, figsize=(8, 6))
    ax = _ax.flatten()

    for i, (var, aux_line, median_line, title) in enumerate(
            zip([ana_ori.var_abs_1d, ana_qua.var_abs_1d, ana_qua.var_binary],
                [False, False, True],
                [False, True, True],
                [f"{name} prequantized: variance in mean-abs -- enc-dist", f"{name} quant-code : variance in mean-abs -- enc-dist",
                 f"{name} quant-code: binary variance -- enc-dist"]
                )):
        liney, intercept, coef, mse, r2 = do_linear_regression(xaxis_range[1:], var[1:])
        ax[i].plot(xaxis_range[1:end_at], var[1:end_at], marker='.', ls='none')
        if aux_line:
            ax[i].plot([1, end_at], [1, 1], color='black', label="y = 1")
        if median_line:
            median = np.median(var)
            ax[i].plot([1, end_at], [median, median], color='black', ls='--', label=f"y median = {median}")
        ax[i].plot(xaxis_range[1:end_at], liney[1:end_at], label=helper_fmt_label(intercept, coef, r2), lw=2)
        ax[i].set_title(title, fontsize=14)
        ax[i].legend()

    fig.tight_layout()
    if show_plot:
        plt.show()
    file_name = os.path.join(os.getcwd(), folder_name, name + ".png")
    fig.savefig(file_name)
    plt.close()


def process_plot_5fig(ana_ori: AnalyzeSmoothnessWholeDatum,
                      ana_qua: AnalyzeSmoothnessWholeDatum,
                      end_at: int, folder_name: str, name: str,
                      plot=True,
                      show_plot=False
                      ):
    xaxis_range = np.arange(len(ana_ori.var_abs_1d))
    if end_at > len(xaxis_range):
        print("ending too large, adjusting...")
        end_at = len(xaxis_range)

    if not plot:
        return None

    fig, _ax = plt.subplots(nrows=3, ncols=2, figsize=(16, 8), sharex='all')
    ax = _ax.flatten()

    for i, (var, aux_line, median_line, title, cr_str) in enumerate(
            zip([
                ana_ori.var_abs_2d,
                ana_ori.var_abs_1d,
                ana_qua.var_abs_2d,
                ana_qua.var_abs_1d,
                None,
                ana_qua.var_binary],
                [False, False, False, False, None, True],
                [False, False, True, True, None, True],
                [
                    f"{name}, PreQuantized: 2Dvar in mean-abs -- enc-dist",
                    f"{name}, PreQuantized: 1Dvar in mean-abs -- enc-dist",
                    f"{name} Quant-Code: 2Dvar in mean-abs -- enc-dist",
                    f"{name} Quant-Code: 1Dvar in mean-abs -- enc-dist",
                    None,
                    f"qua {name}: binary var -- enc-dist"],
                [
                    None, None, None, None, None, True
                ]
            )):
        # skip the 5th position
        if var is None:
            fig.delaxes(ax[i])
            continue

        liney, intercept, coef, mse, r2 = do_linear_regression(xaxis_range[1:], var[1:])
        ax[i].plot(xaxis_range[1:end_at], var[1:end_at], marker='.', ls='none')
        if aux_line:
            ax[i].plot([1, end_at], [1, 1], color='black', label="y = 1")
        if median_line:
            median = np.median(var)
            if i == 5:
                print(f"median of binary var:\t{median}")
            ax[i].plot([1, end_at], [median, median], color='black', ls='--', label=f"y median = {median}")
        ax[i].plot(xaxis_range[1:end_at], liney[1:end_at], label=helper_fmt_label(intercept, coef, r2), lw=2)
        if cr_str:
            p1 = ana_qua.c13y.most_likely_prob
            cr_lb = ana_qua.c13y.huff_est.cr_lb
            cr_ub = ana_qua.c13y.huff_est.cr_ub
            aux_str = ""
            if p1 >= 0.4:
                aux_str = f"CR: {cr_lb:.3f}--{cr_ub:.3f}"
            else:
                aux_str = f"CR: > {cr_lb:.3f}"
            ax[i].set_title("; ".join((title, aux_str)), fontsize=14)
        else:
            ax[i].set_title(title, fontsize=14)
        ax[i].legend()

    fig.tight_layout()
    if show_plot:
        plt.show()
    file_name = os.path.join(os.getcwd(), folder_name, name + ".png")
    fig.savefig(file_name)
    plt.close()

    

def process_plot_new3fig(ana_ori: AnalyzeSmoothnessWholeDatum,
                      ana_qua: AnalyzeSmoothnessWholeDatum,
                      end_at: int, folder_name: str, name: str,
                      plot=True,
                      show_plot=False
                      ):
    xaxis_range = np.arange(len(ana_ori.var_abs_1d))
    if end_at > len(xaxis_range):
        print("ending too large, adjusting...")
        end_at = len(xaxis_range)

    if not plot:
        return None

    fig, _ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 5), sharex='all')
    ax = _ax.flatten()

    i = 0
    for dummy_i, (var, aux_line, median_line, title, cr_str) in enumerate(
            zip([
                None,
                ana_ori.var_abs_1d,
                None,
                ana_qua.var_abs_1d,
                None,
                ana_qua.var_binary],
                [False, False, False, False, None, True],
                [False, False, True, True, None, True],
                [
                    None,
                    f"{name} PreQuantized Original: 1D Variance in Mean Abs Difference",
                    None,
                    f"{name} Quant-Code: 1D Variance in Mean Abs Difference",
                    None,
                    f"{name} Quant-Code: Binary Variance"],
                [
                    None, None, None, None, None, True
                ]
            )):
        # skip position 1,3,5
        if var is None:
#             fig.delaxes(ax[i])
            continue

        liney, intercept, coef, mse, r2 = do_linear_regression(xaxis_range[1:], var[1:])
        ax[i].plot(xaxis_range[1:end_at], var[1:end_at], marker='.', color="#0000EB", ls='none')
        if aux_line:
            ax[i].plot([1, end_at], [1, 1], color='black', label="y = 1")
        if median_line:
            median = np.median(var)
            if i == 5:
                print(f"median of binary var:\t{median}")
            ax[i].plot([1, end_at], [median, median], color='gray', ls='--', label=f"y median = {median}")
        ax[i].plot(xaxis_range[1:end_at], liney[1:end_at], label=helper_fmt_label(intercept, coef, r2), color="#f1c40f", lw=2)
#         if cr_str:
#             p1 = ana_qua.c13y.most_likely_prob
#             cr_lb = ana_qua.c13y.huff_est.cr_lb
#             cr_ub = ana_qua.c13y.huff_est.cr_ub
#             aux_str = ""
#             if p1 >= 0.4:
#                 aux_str = f"CR: {cr_lb:.3f}--{cr_ub:.3f}"
#             else:
#                 aux_str = f"CR: > {cr_lb:.3f}"
# #             ax[i].set_title("; ".join((title, aux_str)), fontsize=14)
#             ax[i].set_title("; ".join((title, aux_str)), fontsize=14)
#         else:
        ax[i].set_title(title, fontsize=14)
        ax[i].legend(fontsize=10)
        i+=1
        
    ax[2].set_xlabel("Encoding Distance", fontsize=10)

    fig.tight_layout()
    if show_plot:
        plt.show()
    file_name = os.path.join(os.getcwd(), folder_name, name + ".png")
    fig.savefig(file_name)
    plt.close()

