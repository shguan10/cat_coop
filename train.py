import pdb


class Train:
    """Performs the training of ``model`` given a training dataset data
    loader, the optimizer, and the loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to train.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - optim (``Optimizer``): The optimization algorithm.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """

    def __init__(self, model, data_loader, optim, criterion, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.metric = metric
        self.device = device

    def run_epoch(self, iteration_loss=False, trainmal=False):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        self.model.train()
        epoch_loss = 0.0
        epoch_dist = 0.0
        epoch_vloss = 0.0
        self.metric.reset()
        numdata = 0
        for step, (inputs,labels) in enumerate(self.data_loader):
            numdata += len(inputs)
            if not trainmal:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

                # Backpropagation
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                epoch_loss += loss.item()

                self.metric.add(outputs.detach(), labels.detach())
            else:
                inputs = inputs.to(self.device)
                l = labels.to(self.device)
                labels = (inputs,l)

                outputs = self.model(inputs)

                dist,vloss = self.criterion(outputs, labels)
                # loss = self.criterion(outputs,inputs)
                loss = .1*dist - 100*vloss
                # loss = -dist

                # Backpropagation
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                epoch_dist += dist.detach()
                epoch_vloss += vloss.detach()
                epoch_loss += loss.detach()

                self.metric.add(outputs[1].detach(), labels[1].detach())

            if iteration_loss: print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))
        print("dist"+str(epoch_dist/numdata))
        print("vloss"+str(epoch_vloss/numdata))
        return epoch_loss / numdata, self.metric.value()
