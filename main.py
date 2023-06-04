import argparse

import torch

from server import Server


def get_args():
    parser = argparse.ArgumentParser(description="Pose Estimation")

    parser.add_argument("--model", type=str, default="./model/fcpose_res50.pth",
                        help="Pretrained model for pose estimation")
    parser.add_argument("--device", type=str, help="Device for prediction, default: cuda,if available")
    parser.add_argument("--port", type=int, default=6006, help="The port for the server")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    model = torch.load(args.model)
    server = Server(model, args.device)
    server.run(host='0.0.0.0', port=args.port, debug=True, allow_unsafe_werkzeug=True)
