from torch.distributions import Normal

def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()

def identity(x):
    return x