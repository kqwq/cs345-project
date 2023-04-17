#!/bin/bash

# Create list of filesnames in the /images directory
filename=$(ls images)

# Loop through each file in the /images directory
for file in $filename; do
  echo "Running SqueezeNet 1.0 on $file"
  python3 run_model.py -i images/$file -m squeezenet1_0

  echo "Running SqueezeNet 1.1 on $file"
  python3 run_model.py -i images/$file -m squeezenet1_1

  echo "Running ResNet 18 on $file"
  python3 run_model.py -i images/$file -m resnet18

  echo "Running AlexNet on $file"
  python3 run_model.py -i images/$file -m alexnet
done
