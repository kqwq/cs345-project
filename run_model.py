import sys
import getopt
from PIL import Image
from torchvision import transforms
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main(argv):
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
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    top5_prob, top5_catid = torch.topk(probabilities, 5)

    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())


if __name__ == "__main__":
    main(sys.argv[1:])
