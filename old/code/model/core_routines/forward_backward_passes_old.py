import numpy as np
import torch


# TODO1: predict only mode: no loss, but key metric: MetricsAggregator
# TODO2: put forward pass train and val in one func, e.g. by adding scope arg > network's mode and suffix deferred; key difficulty is contextmanager with torch.nograd..



def _predict_batch(loss_function, network, data, labels, additional_performance_metrics):
    # additional performance metrics are a dict where a key maps to a function that takes overfit_single_sample and labels as input
    predictions = network(data)
    loss = loss_function(predictions, labels)


    #loss_CPU = loss.data.cpu().numpy()

    additional_performance_metrics_batch_results = dict()
    if additional_performance_metrics:
        for m in additional_performance_metrics:
            result = additional_performance_metrics[m](**{'logits'=predictions, 'labels':labels})
            try:
                result_value = result.data.cpu().numpy()  # in case the metric is calculated on GPU
            except:
                pass
            additional_performance_metrics_batch_results[m] = result_value
    return loss, loss_CPU, additional_performance_metrics_batch_results


#############################
# train forward-backward-pass
#############################

def forward_pass_train(data_loader, optimizer, network, loss_function, additional_performance_metrics, config):
    network.train()  # set to training mode, this affects BN and Dropout
    epoch_loss = np.float(0)
    epoch_additional_performance_metrics = {k: np.float(0) for k in additional_performance_metrics}

    # to avoid overflow, keep running stats
    multiplier = 1 / len(data_loader)

    for step, (data, labels) in enumerate(data_loader):  # i.e. for each batch
        optimizer.zero_grad()  # # reset gradients

        if torch.cuda.is_available():
            data = data.cuda()
            labels = labels.type(torch.LongTensor).cuda()  # (shape: (batch_size, img_h, img_w))

        loss_GPU, loss_CPU, additional_results = _predict_batch(loss_function, network, data, labels,
                                                                additional_performance_metrics)
        epoch_loss += multiplier * loss_CPU
        for k in additional_performance_metrics:
            epoch_additional_performance_metrics[k] += multiplier * additional_results[k]

        # optimization step:
        loss_GPU.backward()  # compute gradients
        optimizer.step()  # perform optimization step

    if config['debug']:
        print(f'Averaged epoch loss on train: {epoch_loss}')
        for m in additional_performance_metrics:
            print(f'Averaged epoch {m} on train: {epoch_additional_performance_metrics[m]}')

    returned_dict = {'loss':epoch_loss, **epoch_additional_performance_metrics}
    suffix = '_train' # add suffix to help Tune distinguish
    returned_dict_w_suffix = {k+suffix:returned_dict[k] for k in returned_dict}

    return returned_dict_w_suffix


##################
# val forward-pass
##################


def forward_pass_val(data_loader, network, loss_function, additional_performance_metrics, config):
    network.eval() # no backward pass, dropout and BN layers freezed
    epoch_loss = np.float(0)
    epoch_additional_performance_metrics = {k: np.float(0) for k in additional_performance_metrics}

    # to avoid overflow, keep running stats
    multiplier = 1/len(data_loader)

    with torch.no_grad():
        # TODO: for inference we could increase the batch size; we could find the max possible size with a hook in the very first iteration

        for step, (data, labels) in enumerate(data_loader):  # i.e. for each batch

            if torch.cuda.is_available():
                data = data.cuda()
                labels = labels.type(torch.LongTensor).cuda()  # (shape: (batch_size, img_h, img_w))

            _, loss_CPU, additional_results = _predict_batch(loss_function, network, data, labels,
                                                             additional_performance_metrics)
            epoch_loss += multiplier * loss_CPU
            for k in additional_performance_metrics:
                epoch_additional_performance_metrics[k] += multiplier * additional_results[k]

    if config['debug']:
        print(f'Averaged epoch loss on val: {epoch_loss}')
        for m in additional_performance_metrics:
            print(f'Averaged epoch {m} on val: {epoch_additional_performance_metrics[m]}')


    returned_dict = {'loss':epoch_loss, **epoch_additional_performance_metrics}
    suffix = '_val' # add suffix to help Tune distinguish
    returned_dict_w_suffix = {k+suffix:returned_dict[k] for k in returned_dict}

    return returned_dict_w_suffix
