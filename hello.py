import torch
from torch import nn


# learning visual features
# FCN (Fully Connected Neural Network)
# - img -> flatten (1d) => spatial information lost
# - huge parameters

class MNISTModel(nn.Module):
	def __init__(self, input_shape: int, hidden_unit: int, output_shape:int):
		super().__init__()
		self.block_1 = nn.Sequential(
			nn.Conv2d(in_channels=input_shape,
					out_channels=hidden_unit,
					kernel_size=3,
					stride=1,
					padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(hidden_unit) # num of parameters in batchnorm2d ?? num_features (int) – channel  
		)

		self.block_2 = nn.Sequential(
			nn.Conv2d(in_channels = hidden_unit,
					out_channels = hidden_unit,
					kernel_size = 3,
					stride = 1,
					padding = 1),	
			nn.ReLU(),
			nn.BatchNorm2d(hidden_unit)
		)

		self.out = nn.Linear(in_features=hidden_unit*7*7, 
					   out_features=output_shape)

	def forward(self, x: torch.Tensor):
		x = self.block_1(x)
		x = self.block_2(x)
		x = self.out(x)
		return x	

def main():
	print(f"Pytorch version: {torch.__version__}")	
	model_0 = MNISTModel(1, 16, 10)
	print(model_0)

if __name__ == "__main__":
	main()
