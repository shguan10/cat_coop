class Malicious_Autoencoder(nn.Module):
  def __init__(self,bboxmodel,trainvictim=True):
    """
    @param bboxmodel : torch.nn.Module
      the victim model to be treated as a black box
    @param trainvictim : bool
      True -- freeze the attacker, train the victim
      False-- freeze the victim, train the attacker
    """
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

  def whichtrain(self,trainvictim=True):
    """
    @param trainvictim : bool
      True -- freeze the attacker, train the victim
      False-- freeze the victim, train the attacker
    """
    if trainvictim: 
      for p in self.parameters(): p.requires_grad = False
      for p in self.bboxmodel.parameters(): p.requires_grad = True
    else: 
      for p in self.bboxmodel.parameters(): p.requires_grad = False
      for p in self.parameters(): p.requires_grad = True

  def transformx(self,x):
    """
    @param x : PyTorch Tensor
      data to transform
    """
    x = self.encode(x)
    x = self.decode(x)
    x = torch.min(x,self.one)
    x = torch.max(x,self.zero)
    return x

  def forward(self,x):
    """
    @param x : PyTorch Tensor
      data to forward through the attacker and the victim
    """
    t = self.transformx(x)
    return (t,self.bboxmodel(t))

  def gettransformeddata(self,xvectors):
    """
    @param xvectors : PyTorch Tensor
      data to transform
    """
    with torch.no_grad():
      data = self.transformx(xvectors)
    print(torch.mean((data-xvectors)**2))
    print(torch.dist(data,xvectors))
    return data
