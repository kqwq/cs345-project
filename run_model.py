import sys
import getopt
from PIL import Image
from torchvision import transforms
import torch
import warnings
from torch.profiler import profile, record_function, ProfilerActivity
import time

# Disable PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main(argv):

    printList = []

    def printLater(*args, **kwargs):
        printList.append((args, kwargs))

    def printAll():
        for args, kwargs in printList:
            print(*args, **kwargs)

    time_start = time.time()
    input_file = ''
    model_name = ''
    help_msg = 'python script.py -i <inputfile> -m <modelname>'
    try:
        opts, args = getopt.getopt(argv, "hi:m:", ["ifile=", "model="])
    except getopt.GetoptError:
        print(help_msg)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_msg)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-m", "--model"):
            model_name = arg

    if not input_file or not model_name:
        print(help_msg)
        sys.exit(2)

    printLater("================ Model Inference ================")
    printLater("- Model: ", model_name)
    printLater("- Input file: ", input_file)
    printLater()
    model = torch.hub.load('pytorch/vision:v0.10.0',
                           model_name, pretrained=True, verbose=False)
    model.eval()

    input_image = Image.open(input_file)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        printLater("Using GPU")
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # Record the memory usage
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                output = model(input_batch)
    profileOutput = prof.key_averages().table(
        row_limit=1)
    row3 = profileOutput.split("\n")[3]
    memoryUsage = row3[117:117+14].strip()

    # # Record CPU time
    # row5 = profileOutput.split("\n")[5]
    # cpuTime = row5[20:].strip()
    # print("CPU time: ", cpuTime)

    # Print top 5 predictions
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    top5_prob, top5_catid = torch.topk(probabilities, 5)

    printLater("#".ljust(5), "Label".ljust(25), "Confidence".ljust(15))
    printLater("-" * 50)
    for i in range(top5_prob.size(0)):
        printLater(str(i+1).ljust(5), categories[top5_catid[i]].ljust(25),
                   str(round(top5_prob[i].item(), 4)).ljust(15))

    time_end = time.time()
    timeDiffMs = (time_end - time_start) * 1000
    printLater("Memory usage: ", memoryUsage)
    printLater("Interface time: ", round(timeDiffMs, 2), "ms")
    printLater()
    printLater()
    printAll()


if __name__ == "__main__":
    main(sys.argv[1:])
