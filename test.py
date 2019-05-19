import torch

import pdb

class Test:
    """Tests the ``model`` on the specified test dataset using the
    data loader, and loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to test.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.
    """

    def __init__(self, model, data_loader, criterion, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.metric = metric
        self.device = device

    def run_epoch(self, iteration_loss=False,trainmal=False):
        """Runs an epoch of validation.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.
        - trainmal (``bool``, optional): If true, tests the network with the two-task loss
                                         else, tests using the normal loss

        Returns:
        - The epoch loss (float), and the values of the specified metrics

        """
        self.model.eval()
        epoch_loss = 0.0
        epoch_dist = 0.0
        epoch_vloss = 0.0
        self.metric.reset()
        numdata = 0
        for step, batch_data in enumerate(self.data_loader):
            numdata += len(batch_data[0])
            if not trainmal:
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                with torch.no_grad():
                    # Forward propagation
                    outputs = self.model(inputs)

                    # Loss computation
                    if iteration_loss: 
                        # pdb.set_trace()
                        print(outputs)
                        print(labels)

                    loss = self.criterion(outputs, labels)

                # Keep track of loss for current epoch
                epoch_loss += loss.item()

                # Keep track of evaluation the metric
                self.metric.add(outputs.detach(), labels.detach())
            else:
                inputs = batch_data[0].to(self.device)
                l = batch_data[1].to(self.device)
                labels = (inputs,l)

                with torch.no_grad():
                    # Forward propagation
                    outputs = self.model(inputs)

                    # Loss computation
                    if iteration_loss: 
                        # pdb.set_trace()
                        print(outputs)
                        print(labels)

                    dist,vloss = self.criterion(outputs, labels)
                    loss = .1*dist - 100*vloss

                epoch_loss += loss.item()

                epoch_dist += dist.detach()
                epoch_vloss += vloss.detach()

                # Keep track of evaluation the metric
                self.metric.add(outputs[1].detach(), labels[1].detach())


            if iteration_loss: print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))
        print("dist"+str(epoch_dist/numdata))
        print("vloss"+str(epoch_vloss/numdata))
        return epoch_loss / numdata, self.metric.value()
