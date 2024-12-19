from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def load_data(self, file_path):
        pass

    @abstractmethod
    def select_variables(self, data):
        pass

    @abstractmethod
    def train(self, data, target, predictors):
        pass

    @abstractmethod
    def get_summary(self):
        pass

    @abstractmethod
    def plot_results(self, data, target, predictors):
        pass

    @abstractmethod
    def predict(self, model, file_path):
        pass
