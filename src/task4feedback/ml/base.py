from torch.nn import Module
from torchrl.modules import ProbabilisticActor, ValueOperator


class ActorCriticModule(Module):
    def __init__(self, actor: Module | ProbabilisticActor, critic: Module | ValueOperator, disciminator: Module = None):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.discriminator = disciminator

    def forward(self, x):
        action = self.actor(x)
        value = self.critic(x)
        return action, value

    def act(self, x):
        return self.actor(x)

    def value(self, x):
        return self.critic(x)
