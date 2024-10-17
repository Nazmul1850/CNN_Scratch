from abc import ABC, abstractmethod

class Layer(ABC):
    def  init_parameters(self):
        pass

    @abstractmethod
    def forward_prop(self, prev_layer):
        pass
    
    @abstractmethod
    def backward_prop(self, dz):
        pass
    
    def update_parameters(self):
        pass
    
    def save_parameters(self):
        pass
    
    def set_parameters(self):
        pass
