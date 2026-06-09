from abc import ABC,abstractmethod

class FeatureProcessor(ABC):

    def fit(self,series):
        return self
    @abstractmethod
    def transform(self,series):
        pass

    def fit_transform(self,series):
        self.fit(series)
        self.transform(series)

