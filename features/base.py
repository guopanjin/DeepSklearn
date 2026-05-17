from abc import ABC,abstractmethod

class FeatureProcessor(ABC):

    def fit(self,series):
        return self
    @abstractmethod
    def transformer(self,series):
        pass

    def fit_transformer(self,series):
        self.fit(series)
        self.transformer(series)

