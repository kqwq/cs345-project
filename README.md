
# cs345-project

Title: Benchmarking PyTorch Vision Models on a Raspberry Pi

## Hardware requirements
A system that can run Python and PyTorch. PyTorch requires a 64-bit operating system.

## Run locally

Step 1
Clone this repository with `git clone https://kqwq/cs345-project.git`

Step 2
Run `init.sh`. If you don't have Python 3 on your machine, install it.

Step 3
Run `benchmark.sh -o NAME_OF_HARDWARE.txt` to log the benchmark results. To use your own images, replace the images in the `images` folder with your own images.
Replace the `NAME_OF_HARDWARE` part with the name of the device, such as "Raspberry_Pi_3B".
