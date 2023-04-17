#/bin/bash

echo "Running SqueezeNet 1.0"
python3 run_model.py -i dog.jpg -m squeezenet1_0

echo "Running SqueezeNet 1.1"
python3 run_model.py -i dog.jpg -m squeezenet1_1

echo "Running ResNet 18"
python3 run_model.py -i dog.jpg -m resnet18

echo "Running AlexNet"
python3 run_model.py -i dog.jpg -m alexnet
