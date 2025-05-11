import argparse
import torch
from ufld.model.model import parsingNet
from ufld.utils.common import merge_config
from model import faster_rcnn
from get_video import run_video, run_video_file

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for lane detection and sign detection")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output video file")
    return parser.parse_args()

def main():
    args = parse_args()
    # Load model
    sign_model = faster_rcnn(num_classes=43)  # GTSRB có 43 lớp
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("checkpoint_best.pth", map_location=device)
    sign_model.load_state_dict(checkpoint["model_state_dict"])
    sign_model.to(device)
    sign_model.eval()

    cfg, _ = merge_config()
    lane_model = parsingNet(pretrained=False, backbone='18', cls_dim=(200+1, 18, 4), use_aux=False).to(device)
    state_dict = torch.load('ufld/models/culane_18.pth', map_location=device)
    lane_model.load_state_dict(state_dict['model'])
    lane_model.eval()

    # run_video(
    #     model=model,
    #     frames_folder="video_example/05081544_0305",
    #     output_video_path="output_result.mp4",
    #     device=device,
    #     threshold=0.5
    # )

    run_video_file(
        detection_model=sign_model,
        lane_model=lane_model,
        input_video_path=args.video,
        output_video_path=args.output,
        device=device,    
    )

if __name__ == "__main__":
    main()