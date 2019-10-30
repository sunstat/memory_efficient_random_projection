import os
import matplotlib.pyplot as plt
import numpy as np

dir_name, _ = os.path.split(os.path.abspath(__file__))
fig_base = os.path.join(dir_name, 'figures')

markers = ['o', 'v', '*']
colors = ['r', 'g', 'b']

def plot(tensor_dim, res, fig_name):
    '''
    generate two plots, one for dense tensor, the other for low rank square tensor;
    '''
    def _collect_data(res, rp_method, vr_method, tensor_type):
        curr_res = dict()
        curr_res['val'] = []
        curr_res['upper_bound'] = []
        curr_res['lower_bound'] = []

        for target_k in res.keys():
            vals = res[target_k][vr_method][tensor_type][rp_method]
            curr_res['val'].append(np.mean(vals))
            curr_res['upper_bound'].append(np.quantile(vals, 0.975))
            curr_res['lower_bound'].append(np.quantile(vals, 0.025))
        print(curr_res)
        return curr_res

    def _plot(relative_err_curves, x, tensor_type):
        fig, ax = plt.subplots()
        # plot val
        for i, vr_method in enumerate(['raw', 'geomedian', 'average']):
            for rp_method in ['TRP', 'normal']:
                label = '{}-{}'.format(rp_method, '' if vr_method == 'raw' else 'vr-{}'.format(vr_method))
                key = '_'.join([rp_method, vr_method, tensor_type])
                ax.plot(x, relative_err_curves[key]['val'], color=colors[i], label=label, marker=markers[i])
                ax.plot(x, relative_err_curves[key]['upper_bound'], color=colors[i], linestyle=':', marker=markers[i])
                ax.plot(x, relative_err_curves[key]['lower_bound'], color=colors[i], linestyle=':', marker=markers[i])
        return fig

    res = res[tensor_dim]
    x = list(res.keys())
    # two random projection methods, three variance reduction methods, two type of tensors
    relative_err_curves = dict()
    for rp_method in ['TRP', 'normal']:
        for vr_method in ['raw', 'geomedian', 'average']:
            for tensor_type in ['dense_tensor', 'lr_tensor']:
                curve_cfg = '_'.join([rp_method, vr_method, tensor_type])
                relative_err_curves[curve_cfg] = _collect_data(res, rp_method, vr_method, tensor_type)
    
    fig1 = _plot(relative_err_curves, x, 'dense_tensor')
    fig1.savefig(os.path.join(fig_base, '{}_dense.png'.format(fig_name)))
    plt.close(fig1)
    fig2 = _plot(relative_err_curves, x, 'lr_tensor')
    fig2.savefig(os.path.join(fig_base, '{}_lrank.png'.format(fig_name)))
    plt.close(fig2)
