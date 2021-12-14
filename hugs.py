import argparse
from datetime import datetime

from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.TorchVisionClassifierTrainer import TorchVisionClassifierTrainer

parser = argparse.ArgumentParser(description='Image classifier')
parser.add_argument('--imgs', type=str, default="./sharks/", help='The directory of the input images')
parser.add_argument('--output', type=str, default="./OUT_TORCHVISION/", help='The output directory of the model')
parser.add_argument('--model', type=str, default="densenet121", help='The TorchVision model')
parser.add_argument('--epochs', type=int, default=50, help='Number of Epochs')
args = parser.parse_args()

# Load the dataset
train, test, id2label, label2id = VisionDataset.fromImageFolder(
	args.imgs,
	test_ratio=0.10,
	balanced=True,
	torch_vision=True,
)

# Train the model
trainer = TorchVisionClassifierTrainer(
	output_dir   = args.output + str(datetime.today().strftime("%Y-%m-%d-%H-%M-%S")) + "/",
	model_name   = args.model,
	train      	 = train,
	test      	 = test,
	batch_size   = 64,
	max_epochs   = args.epochs,
	id2label 	 = id2label,
	label2id 	 = label2id,
	lr=1e-3,
)

