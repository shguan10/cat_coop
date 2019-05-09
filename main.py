import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms

import transforms as ext_transforms
from models.enet import ENet
from train import Train
from test import Test
from metric.iou import IoU
from args import get_arguments
from data.utils import enet_weighing, median_freq_balancing
import utils

import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle as pk

# Get the arguments
args = get_arguments()

if args.device=='cuda': device = torch.device(args.device)
else: device = torch.device('cpu')


def load_dataset(dataset,cached=False):
    print("\nLoading dataset...\n")

    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

    image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])

    label_transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        ext_transforms.PILToLongTensor()
    ])

    # Get selected dataset
    # Load the training set as tensors
    train_set = dataset(
        args.dataset_dir,
        transform=image_transform,
        label_transform=label_transform)
    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    # Load the validation set as tensors
    val_set = dataset(args.dataset_dir,
                      mode='val',
                      transform=image_transform,
                      label_transform=label_transform)
    val_loader = data.DataLoader(val_set,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.workers)

    # Load the test set as tensors
    test_set = dataset(
        args.dataset_dir,
        mode='test',
        transform=image_transform,
        label_transform=label_transform)
    test_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)

    # Get encoding between pixel valus in label images and RGB colors
    class_encoding = train_set.color_encoding

    # Remove the road_marking class from the CamVid dataset as it's merged
    # with the road class
    if args.dataset.lower() == 'camvid':
        del class_encoding['road_marking']

    # Get number of classes to predict
    num_classes = len(class_encoding)

    # Print information for debugging
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    # Get a batch of samples to display
    if args.mode.lower() == 'test':
        images, labels = next(iter(test_loader))
    else:
        images, labels = next(iter(train_loader))
    print("Image size:", images.size())
    print("Label size:", labels.size())
    print("Class-color encoding:", class_encoding)

    # Show a batch of samples and labels
    if args.imshow_batch:
        print("Close the figure window to continue...")
        label_to_rgb = transforms.Compose([
            ext_transforms.LongTensorToRGBPIL(class_encoding),
            transforms.ToTensor()
        ])
        color_labels = utils.batch_transform(labels, label_to_rgb)
        utils.imshow_batch(images, color_labels)

    # Get class weights from the selected weighing technique
    print("\nWeighing technique:", args.weighing)
    print("Computing class weights...")
    print("(this can take a while depending on the dataset size)")
    
    if not cached:
        class_weights = 0
        if args.weighing.lower() == 'enet':
            class_weights = enet_weighing(train_loader, num_classes)
        elif args.weighing.lower() == 'mfb':
            class_weights = median_freq_balancing(train_loader, num_classes)
        else:
            class_weights = None

        with open("tmp.pk","wb") as f: pk.dump(class_weights,f)
    else:
        with open("tmp.pk","rb") as f: class_weights = pk.load(f)


    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(device)
        # Set the weight of the unlabeled class to 0
        if args.ignore_unlabeled:
            ignore_index = list(class_encoding).index('unlabeled')
            class_weights[ignore_index] = 0

    print("Class weights:", class_weights)

    return (train_loader, val_loader, test_loader), class_weights, class_encoding

def train(train_loader, val_loader, class_weights, class_encoding, 
            pretrained="/home/xinyu/work/PyTorch-ENet/save/ENet.pt"):
    print("\nTraining...\n")

    num_classes = len(class_encoding)

    model = ENet(num_classes)

    if pretrained: 
        model.load_state_dict(torch.load(pretrained)["state_dict"])

    # Intialize ENet
    model = model.to(device)
    # Check if the network architecture is correct
    print(model)

    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ENet authors used Adam as the optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    # Learning rate decay scheduler
    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs,
                                     args.lr_decay)

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # Optionally resume from a checkpoint
    if args.resume:
        model, optimizer, start_epoch, best_miou = utils.load_checkpoint(
            model, optimizer, args.save_dir, args.name)
        print("Resuming from model: Start epoch = {0} "
              "| Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0

    # Start Training
    train = Train(model, train_loader, optimizer, criterion, metric, device)
    val = Test(model, val_loader, criterion, metric, device)
    for epoch in range(start_epoch, args.epochs):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        lr_updater.step()
        epoch_loss, (iou, miou) = train.run_epoch(args.print_step)

        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
              format(epoch, epoch_loss, miou))

        if epoch % 10 == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            loss, (iou, miou) = val.run_epoch(args.print_step)

            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
                  format(epoch, loss, miou))

            # Print per class IoU on last epoch or if best iou
            if epoch + 1 == args.epochs or miou > best_miou:
                for key, class_iou in zip(class_encoding.keys(), iou):
                    print("{0}: {1:.4f}".format(key, class_iou))

            # Save the model if it's the best thus far
            if miou > best_miou:
                print("\nBest model thus far. Saving...\n")
                best_miou = miou
                utils.save_checkpoint(model, optimizer, epoch + 1, best_miou,
                                      args)

    return model


def test(model, test_loader, class_weights, class_encoding):
    print("\nTesting...\n")
    num_classes = len(class_encoding)


    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # Test the trained model on the test set
    test = Test(model, test_loader, criterion, metric, device)

    print(">>>> Running test dataset")

    loss, (iou, miou) = test.run_epoch(iteration_loss=False)
    class_iou = dict(zip(class_encoding.keys(), iou))

    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

    # Print per class IoU
    for key, class_iou in zip(class_encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou))

    # Show a batch of samples and labels
    # if args.imshow_batch:
    if True:
        print("A batch of predictions from the test set...")
        images, _ = next(iter(test_loader))
        predict(model, images, class_encoding)


def predict(model, images, class_encoding):
    images = images.to(device)

    # Make predictions!
    model.eval()
    with torch.no_grad():
        predictions = model(images)

    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    _, predictions = torch.max(predictions.data, 1)

    label_to_rgb = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
    utils.imshow_batch(images.data.cpu(), color_predictions)

class Malicious_Autoencoder(nn.Module):
  def __init__(self,bboxmodel,trainvictim=True):
    nn.Module.__init__(self)
    self.encode = nn.Conv2d(3,8,3,stride=2,padding=1,bias=True)
    self.decode = nn.ConvTranspose2d(
                    8,
                    3,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False)
    self.bboxmodel = bboxmodel
    self.one = torch.tensor(1.).to(device)
    self.zero = torch.tensor(0.).to(device)
    self.whichtrain(trainvictim)
    # self.a = torch.tensor(1.,requires_grad=True).to(device)

  def whichtrain(self,trainvictim=True):
    if trainvictim: 
      for p in self.parameters(): p.requires_grad = False
      for p in self.bboxmodel.parameters(): p.requires_grad = True
    else: 
      for p in self.bboxmodel.parameters(): p.requires_grad = False

  def transformx(self,x):
    x = self.encode(x)
    x = self.decode(x)
    x = torch.min(x,self.one)
    x = torch.max(x,self.zero)
    return x
    # return x*self.a

  def forward(self,x):
    t = self.transformx(x)
    return (t,self.bboxmodel(t))

  def gettransformeddata(self,xvectors):
    with torch.no_grad():
      data = self.transformx(xvectors)
    print(torch.mean((data-xvectors)**2))
    print(torch.dist(data,xvectors))
    return data

def trainmal(model, train_loader, val_loader, class_weights, class_encoding,
            pretrained="/home/xinyu/work/PyTorch-ENet/save/mal.pt"):
    print("\nTraining Attacker...\n")

    num_classes = len(class_encoding)

    model = Malicious_Autoencoder(model)

    if pretrained: 
        model.load_state_dict(torch.load(pretrained)["state_dict"])

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)

    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs, args.lr_decay)

    # Evaluation metric
    if args.ignore_unlabeled: ignore_index = list(class_encoding).index('unlabeled')
    else: ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    start_epoch = 0
    best_miou = 0
    best_loss = 99999999

    def new_loss(outputs,labels):
      transformx, bout = outputs
      origx, desireds = labels

      l1 = torch.dist(transformx,origx)
      l2 = criterion(bout,desireds)

      return l1,l2

    # Start Training
    train = Train(model, train_loader, optimizer, new_loss, metric, device)
    val = Test(model, val_loader, new_loss, metric, device)
    for epoch in tqdm(range(start_epoch, args.epochs)):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        lr_updater.step()
        epoch_loss, (iou, miou) = train.run_epoch(args.print_step,trainmal=True)

        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
              format(epoch, epoch_loss, miou))

        if (epoch + 1) % 3 == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            loss, (iou, miou) = val.run_epoch(args.print_step,trainmal=True)

            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
                  format(epoch, loss, miou))

            # Print per class IoU on last epoch or if best iou
            if epoch + 1 == args.epochs or miou > best_miou:
                for key, class_iou in zip(class_encoding.keys(), iou):
                    print("{0}: {1:.4f}".format(key, class_iou))

            # Save the model if it's the best thus far
            # if miou > best_miou:
            if loss < best_loss:
                print("\nBest model thus far. Saving...\n")
                # best_miou = miou
                best_loss = loss
                n = args.name
                args.name = 'mal.pt'
                utils.save_checkpoint(model, optimizer, epoch + 1, best_miou, args)
                args.name = n

    return model

def out2segs(images):
    assert len(images.shape)==4
    return torch.argmax(images,dim=1)

def displaymal(model, train_loader, val_loader, class_weights, class_encoding,
               pretrained="/home/xinyu/work/PyTorch-ENet/save/mal.pt"):
    print("\n Displaying Attacker...\n")

    num_classes = len(class_encoding)

    model = Malicious_Autoencoder(model)

    if pretrained: model.load_state_dict(torch.load(pretrained)["state_dict"])

    model = model.to(device)

    def displaybatch(loader,tag):
        for i in range(10):
            images, labels = next(iter(loader))
            print("Image size:", images.size())
            print("Label size:", labels.size())
            print("Class-color encoding:", class_encoding)

            label_to_rgb = transforms.Compose([
                ext_transforms.LongTensorToRGBPIL(class_encoding),
                transforms.ToTensor()
            ])
            color_labels = utils.batch_transform(labels, label_to_rgb)

            imagescuda = images.to(device)
            vlabels = model.bboxmodel(imagescuda).to('cpu')
            vlabels = out2segs(vlabels)
            vlabels = utils.batch_transform(vlabels,label_to_rgb)

            print("Original...")
            # utils.imshow_batch(images, vlabels)
            img = images[i].numpy().transpose((1,2,0))
            v = vlabels[i].numpy().transpose((1,2,0))

            plt.imsave("orig"+tag+str(i)+".png",img)
            plt.imsave("origseg"+tag+str(i)+".png",v)

            newimages = model.gettransformeddata(imagescuda)
            _,psegs = model(newimages)

            newimages = newimages.to('cpu')
            psegs = out2segs(psegs.to('cpu'))

            psegs = utils.batch_transform(psegs,label_to_rgb)
            img = newimages[i].numpy().transpose((1,2,0))
            v = psegs[i].numpy().transpose((1,2,0))
            plt.imsave("results/trans"+tag+str(i)+".png",img)
            plt.imsave("results/transseg"+tag+str(i)+".png",v)
            # utils.imshow_batch(newimages, psegs)

    print("training images")
    displaybatch(train_loader,"train")
    print("val images")
    displaybatch(val_loader,"val")

def ensemble(model, train_loader, val_loader, class_weights, class_encoding,
               pretrained="/home/xinyu/work/PyTorch-ENet/save/mal.pt"):
    stoptrainmal_thres_thres1 = 0.05 # a recon less than this means we can start using the autoencoder to attack

    stoptrainmal_recon_thres2 = 32 # a recon loss less than this means we can continue to next it

    print("\n Training Ensemble...\n")

    num_classes = len(class_encoding)

    model = Malicious_Autoencoder(model,trainvictim=True)

    if pretrained: model.load_state_dict(torch.load(pretrained)["state_dict"])

    model = model.to(device)



if __name__ == '__main__':

    assert os.path.isdir(args.dataset_dir),\
        "The directory \"{0}\" doesn't exist.".format(args.dataset_dir)

    assert os.path.isdir(args.save_dir),\
        "The directory \"{0}\" doesn't exist.".format(args.save_dir)

    # Import the requested dataset
    if args.dataset.lower() == 'camvid':
        from data import CamVid as dataset
    elif args.dataset.lower() == 'cityscapes':
        from data import Cityscapes as dataset
    else: raise RuntimeError("\"{0}\" is not a supported dataset.".format(args.dataset))

    loaders, w_class, class_encoding = load_dataset(dataset,cached=True)
    train_loader, val_loader, test_loader = loaders

    if args.mode.lower() in {'train', 'full'}:
        model = train(train_loader, val_loader, w_class, class_encoding)
        if args.mode.lower() == 'full':
            test(model, test_loader, w_class, class_encoding)
    elif args.mode.lower() == 'test':
        # Intialize a new ENet model
        num_classes = len(class_encoding)
        model = ENet(num_classes).to(device)

        # Initialize a optimizer just so we can retrieve the model from the
        # checkpoint
        optimizer = optim.Adam(model.parameters())

        # Load the previoulsy saved model state to the ENet model
        model = utils.load_checkpoint(model, optimizer, args.save_dir,
                                      args.name)[0]
        print(model)
        test(model, test_loader, w_class, class_encoding)
    elif args.mode.lower() == 'trainmal':
        # Intialize a new ENet model
        num_classes = len(class_encoding)
        model = ENet(num_classes).to(device)

        # Initialize a optimizer just so we can retrieve the model from the
        # checkpoint
        optimizer = optim.Adam(model.parameters())

        # Load the previoulsy saved model state to the ENet model
        model = utils.load_checkpoint(model, optimizer, args.save_dir, args.name)[0]

        trainmal(model, train_loader, test_loader, w_class, class_encoding)
    
    elif args.mode.lower() == 'dis':
        # Intialize a new ENet model
        num_classes = len(class_encoding)
        model = ENet(num_classes).to(device)

        # Initialize a optimizer just so we can retrieve the model from the
        # checkpoint
        optimizer = optim.Adam(model.parameters())

        # Load the previoulsy saved model state to the ENet model
        model = utils.load_checkpoint(model, optimizer, args.save_dir, args.name)[0]

        displaymal(model, train_loader, test_loader, w_class, class_encoding)

    else:
        raise RuntimeError("\"{0}\" is not a valid choice for execution mode.".format(args.mode))
