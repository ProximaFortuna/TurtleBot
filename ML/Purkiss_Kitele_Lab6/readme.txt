Train resnet can be run with
	python3 {path}/train_resnet.py -p {Path to image folder}

This generates a resnet model yippppeeeeee. It comes in 3 parts, they must be in the same folder I think. Just to be safe probably put them in the same folder the grader script is in.

You can run model grader with
	python3 {path}/model_grader.py --data_path {Path to images} --model_path {Path to folder with model}/signs_resnet18.onnx
	
You need onnx and torch

Should take ~1 minute to train on cpu, could realistically be done on like a tenth of the epochs probably but why not some overkill.

I put the training images in the supplemental folder so it can all be run with just the folder.
