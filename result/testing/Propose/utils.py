import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

def mean_std(distribution):
    mean, std = 0.0, 0.0
    for j, e in enumerate(distribution, 1):
        mean += j * e
    for k, e in enumerate(distribution, 1):
        std += e * (k - mean) ** 2
    return mean, std ** 0.5

def emd(pred, gt, r=1):
    emd_loss = 0.0
    p, q = np.array(pred), np.array(gt)
    length = len(p)
    for i in range(1, length + 1):
        emd_loss += np.abs(sum(p[:i] - q[:i])) ** r
    return (emd_loss / length) ** (1. / r)

def euclid(pred, gt):
    return np.sqrt(np.sum((np.array(pred) - np.array(gt)) ** 2))

def lcc(pred, gt):
    corr, p = stats.pearsonr(pred, gt)
    return corr, p

def srcc(pred, gt):
    corr, p = stats.spearmanr(pred, gt)
    return corr, p

def binary_accuracy(pred, gt):
    count = 0
    for a, b in zip(pred, gt):
        count += (a < 5.0) ^ (b < 5.0)
    return 1 - (count / len(pred))

def get_bins(*args, no_bins = 50):
    m = min([min(x) for x in args])
    M = max([max(x) for x in args])
    return np.linspace(m, M, no_bins)

def create_figure(figrow, figname):
    figure, axeses = plt.subplots(figrow, 2) # 2 columns
    figure.set_figwidth(15)
    figure.set_figheight(figrow * 5)
    figure.suptitle(figname, fontsize=16)
    axeses = axeses.ravel()
    return axeses

def compare_with_ground_truth(data, columns, bins = 50, figname = ''):
    axes = create_figure((len(columns) + 1) // 2, figname)
    for col, ax in zip(columns, axes):
        gt, pred = data[f'gt_{col}'], data[col]
        rb = get_bins(gt, pred, no_bins=bins)
        ax.hist(gt, bins=rb, alpha=0.5)
        ax.hist(pred, bins=rb, alpha=0.5)
        ax.legend(['GT', 'Pred'])
        ax.set_title(col)
    plt.show()

def _summarize(row, metrics: dict):
    pred, gt = row[1:11], row[11:]
    mean, std = mean_std(pred)
    gtmean, gtstd = mean_std(gt)
    diffmean, diffstd = abs(mean - gtmean), abs(std - gtstd)
    output_columns = ['id', 'mean', 'std', 'gt_mean', 'gt_std', 'diff_mean', 'diff_std']
    outputs = [row[0], mean, std, gtmean, gtstd, diffmean, diffstd]
    for name, func in metrics.items():
        output_columns.append(name)
        outputs.append(func(pred, gt))
        
    return pd.Series(outputs, output_columns)

def summarize(*, file = None, data = None, metrics: dict, corrs: dict, output_file: str = None):
    if file:
        data = pd.read_csv(file, header=None)
    data_sum = data.apply(_summarize, axis=1, metrics=metrics)
    print(f"{'Accuracy':<13}: {binary_accuracy(data_sum['mean'], data_sum['gt_mean']) * 100:.2f}%")
    data_sum = data_sum.astype({'id': int})
    for col in data_sum.columns[5:]:
        print(f'{col.upper():<13}: {data_sum[col].mean():.5f} [mean] {data_sum[col].max():.5f} [max] {data_sum[col].min():.5f} [min]')
        
    if output_file:
        data_sum.to_csv(output_file, index=None)
    for name, func in corrs.items():
        print(f'{name.upper():<6} [mean]:', func(data_sum['mean'], data_sum['gt_mean']))
        print(f'{name.upper():<6}  [std]:', func(data_sum['std'], data_sum['gt_std']))
    return data_sum

def _detail(pred: pd.DataFrame, gt:pd.DataFrame, figsize = (6.4, 4.8), title=None):
    plt.figure(figsize=figsize)
    plt.plot(gt, label='GT')
    plt.axhline(5.0, linewidth=3.0, linestyle='--')
    plt.plot(pred, linestyle='solid', label='PRED', marker='o', alpha=1.0)
    plt.legend()
    plt.title(title)

def show_detail(data: pd.DataFrame, col: str, *, title = None, start = None, end = None, count = None, figsize = None):
    filter = data.sort_values(by=f'gt_{col}', ignore_index=True)
    if start:
        filter = filter[filter[f'gt_{col}'] >= start]
    if end:
        filter = filter[filter[f'gt_{col}'] <= end]
    print(f'There are {len(filter)} samples in range [{start}, {end}]')
    if count:
        filter = filter[::max(1, len(filter) // count)]
    _detail(filter[col], filter[f'gt_{col}'], figsize, title)
    return filter

def statistic(named_data: dict[str, pd.DataFrame], column: str, *, start = 2, stop = 10, step=0.5):
    dd = {'score': [''], **{m: [0] for m in named_data}}
    score_range = np.linspace(start, stop, int((stop - start) / step) + 1)
    for i in range(1, len(score_range)):
        dd['score'].append(f'{score_range[i-1]:.1f}-{score_range[i]:.1f}')
        for col in dd:
            if col == 'score': continue
            dd[col].append(len(named_data[col][named_data[col][column] <= score_range[i]]) - sum(dd[col]))
    return pd.DataFrame(dd).iloc[1:]

def plot_trainning(state_path: str, labels: list[str], highlight_on: str = 'min', *,
              savefig: bool = False, figdir = './', title: str = 'summary', vertical_line_at: list[int] = None):
    states = pd.read_csv(state_path, header=None)
    states.columns = labels
    N = len(states)

    _mm = max([len(l) for l in labels])
    if highlight_on == 'min':
        _values = states.min()
        _indexes = states.idxmin()
    else:
        highlight_on = 'max'
        _values = states.max()
        _indexes = states.idxmax()

    for col in labels:
        plt.plot(range(1, N + 1), states[col], label=col)
        plt.axhline(_values[col], linestyle='--')
        print(f'{highlight_on.upper()} [{col.rjust(_mm)}] = {_values[col]:.4f} at {_indexes[col] + 1}')
    plt.title(title)
    plt.legend()
    
    if vertical_line_at is not None:
        for v in vertical_line_at:
            plt.axvline(v, linestyle='--', color='green')

    if savefig:
        plt.savefig(os.path.join(figdir, f'{title}.png'))
