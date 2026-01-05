from abc import ABC, abstractmethod


class Trainer(ABC):
    @abstractmethod
    def observe(self, trajectory):
        pass

    @abstractmethod
    def update(self):
        pass

    def reset(self):
        pass
