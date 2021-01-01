import matplotlib.lines as mlines
import matplotlib.pylab as plt


# TODO: specify epoch interval (x-axis)


def plot_mIoUs_and_losses(d, outpath, model_id=None, dataset_id=None):
    epoch_losses_train = d['train']['loss']
    epoch_losses_val = d['val']['loss']
    epoch_mIoUs_train = d['train']['MetricsAggregator']
    epoch_mIoUs_val = d['val']['MetricsAggregator']

    fig, ax1 = plt.subplots(constrained_layout=False)

    # MetricsAggregator
    # y-axis
    color = 'tab:orange'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MetricsAggregator', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # lines
    ax1.plot(epoch_mIoUs_val, 'b', color=color)
    ax1.plot(epoch_mIoUs_val, 'bo', color=color)
    # performance on train in gray but with same sympols
    ax1.plot(epoch_mIoUs_train, 'b', color='gainsboro')
    ax1.plot(epoch_mIoUs_train, 'bo', color='gainsboro')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # Losses
    # y-axis
    color = 'k'
    ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    # lines
    ax2.plot(epoch_losses_val, 'k', color=color)
    ax2.plot(epoch_losses_val, 'ks', color=color)
    # performance on train in gray but with same sympols
    ax2.plot(epoch_losses_train, 'k', color='gainsboro')
    ax2.plot(epoch_losses_train, 'ks', color='gainsboro')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # legends
    orange_line_mIoU_val = mlines.Line2D([], [], color='orange', marker='o',
                                         markersize=5, label='MetricsAggregator on Val')
    grey_line_mIoU_train = mlines.Line2D([], [], color='gainsboro', marker='o',
                                         markersize=5, label='MetricsAggregator on Train')
    black_line_loss_val = mlines.Line2D([], [], color='k', marker='s',
                                        markersize=5, label='Val Loss')
    grey_line_loss_train = mlines.Line2D([], [], color='gainsboro', marker='s',
                                         markersize=5, label='Train Loss')

    handles = [orange_line_mIoU_val, grey_line_mIoU_train, black_line_loss_val, grey_line_loss_train]
    plt.legend(handles=handles, loc="center left")
    # https://stackoverflow.com/questions/44413020/how-to-specify-legend-position-in-matplotlib-in-graph-coordinates

    plt.title(f"\nTraining and Validation Performance of \nModel {model_id},"
              f"\nDataset {dataset_id}")
    # plt.show()
    plt.tight_layout()
    plt.savefig("%s/all_results_visualized.png" % outpath)

# # test
# from old_utils.old_utils.serialization import dict_from_yaml
# abs_path_to_yaml = '/Users/d071503/Thesis/Results/shared_results/Cityscapes/Test.3/all_results.yaml'
# outpath = '/Users/d071503/Thesis/Results/shared_results/Cityscapes/Test.3/'
# d = dict_from_yaml(abs_path_to_yaml)
# plot_mIoUs_and_losses(d=d,outpath=outpath)
