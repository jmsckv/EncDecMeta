import torch

# TODO: would be nice to condense this in one forward pass taking arg 'train'/'val'
# key difficulty: context manager to deactivate gradient backprop

#############################
# train forward-backward-pass
#############################

def forward_pass_train(data_loader, optimizer, network, loss_function, metrics_aggregator, config):

    network.train()  # set to training mode, affects BN and Dropout
    loss_multiplier = torch.cuda.DoubleTensor([1/len(data_loader)]) if torch.cuda.is_available() else torch.DoubleTensor([1/len(data_loader)])
    epoch_loss = torch.cuda.DoubleTensor([0]) if torch.cuda.is_available() else torch.DoubleTensor([0])

    if config['debug']:
        if torch.cuda.is_available():
            assert loss_multiplier.is_cuda and epoch_loss.is_cuda

    metrics_aggregator.reset_state()

    for step, (data, labels) in enumerate(data_loader): # for each batch...

        if config.get('verbose', False):
            print(f'\nTraining loop, batch {step}.')

        optimizer.zero_grad()  # reset gradients
        if torch.cuda.is_available():
            data = data.cuda()
            labels = labels.type(torch.LongTensor).cuda()
        logits = network(data)
        loss = loss_function(logits, labels)
        epoch_loss += loss_multiplier * loss.detach().data
        metrics_aggregator.update_state(logits,labels)
        loss.backward()  # compute gradients
        optimizer.step()  # perform optimization step

        if config['debug']:
            if torch.cuda.is_available():
                assert loss_multiplier.is_cuda and epoch_loss.is_cuda and loss.is_cuda

    return metrics_aggregator.get_metrics(add_metrics={'loss': epoch_loss.detach()})


##################
# val forward-pass
##################

def forward_pass_val(data_loader, network, loss_function, metrics_aggregator, config):

    network.eval() # no backward pass, dropout and BN layers frozen
    loss_multiplier = torch.cuda.DoubleTensor([1 / len(data_loader)]) if torch.cuda.is_available() else torch.DoubleTensor([1 / len(data_loader)])
    epoch_loss = torch.cuda.DoubleTensor([0]) if torch.cuda.is_available() else torch.DoubleTensor([0])

    if config['debug']:
        if torch.cuda.is_available():
            assert epoch_loss.is_cuda and loss_multiplier.is_cuda

    metrics_aggregator.reset_state()

    with torch.no_grad():
        for step, (data, labels) in enumerate(data_loader):

            if config.get('verbose', False):
                print(f'\nValidation loop, batch {step}.')

            if torch.cuda.is_available():
                data = data.cuda()
                labels = labels.type(torch.LongTensor).cuda()
            logits = network(data)
            loss = loss_function(logits, labels)
            epoch_loss += loss_multiplier * loss.detach().data
            metrics_aggregator.update_state(logits, labels)

    if config['debug']:
        if torch.cuda.is_available():
            assert loss_multiplier.is_cuda and epoch_loss.is_cuda and loss.is_cuda

    return metrics_aggregator.get_metrics(add_metrics={'loss': epoch_loss})

