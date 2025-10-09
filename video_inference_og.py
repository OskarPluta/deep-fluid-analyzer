import sys
import cv2 as cv
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from collections import Counter, deque

sys.path.append(str(Path(__file__).parent / 'src'))
from dataset import get_validation_transforms
from model import create_model, FluidSegmentationModel

from utils import *


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FluidSegmentationModel(
            encoder_name='efficientnet-b3',
            encoder_weights=None,
            in_channels=3,
            classes=1
        )   
    model_state_dict = torch.load('best_model/best_model.pth', map_location=device)["model_state_dict"]
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    base_dir = '/media/oskar/1C81-E822/'

    videos = os.listdir(base_dir)

    videos = [v for v in videos if v.endswith('.MP4')]
    # print(videos)
    # videos = ['/media/oskar/1C81-E822/0,7% M&GK 0,175 3.MP4']
    
    for video in videos:
        video_path = os.path.join(base_dir, video)
        print(video_path)
        cap = cv.VideoCapture(video_path)

        orig_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))) # (w, h)
        print(f'Original video size: {orig_size}')
        image_size = (512, 512) # size for the model, (h,w) for albumenations
        transform = get_validation_transforms(image_size=image_size)

        h_ratio = orig_size[1] / image_size[0]
        w_ratio = orig_size[0] / image_size[1]
        print(f'Height ratio: {h_ratio}, Width ratio: {w_ratio}')

        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))


        end_of_expansion_frame, break_moment_frame = get_start_end_frames(cap,
                                                                        model,
                                                                        device,
                                                                        transform,
                                                                        display=True)
        
        # get the cropping bounds
        cap.set(cv.CAP_PROP_POS_FRAMES, end_of_expansion_frame + 10)
        ret, frame = cap.read()

        pred_mask = get_binary_mask(frame, model, device, transform)
        contours = cv.findContours(pred_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        original_bounds = get_fluid_bounds(contours, fluid_broke=False)
        new_top, new_bottom = crop_fluid_region(frame, original_bounds, w_ratio, h_ratio, 512)
        # end of getting cropping bounds

        new_top, new_bottom = 0, orig_size[1]  # disable cropping for now

        x_expected_value = get_expected_narrowest_point_v1(cap,
                                                        model,
                                                        device,
                                                        transform,
                                                        end_of_expansion_frame,
                                                        break_moment_frame,
                                                        (0, 1080),
                                                        True)

        diameters = []

        cap.set(cv.CAP_PROP_POS_FRAMES, end_of_expansion_frame)

        while True:
            current_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            if break_moment_frame is not None:
                if current_frame >= break_moment_frame:
                    fluid_broke = True
                else:
                    fluid_broke = False
            else:
                fluid_broke = False

            ret, frame = cap.read()

            cropped_frame = frame[new_top:new_bottom, :]

            cropped_mask = get_binary_mask(cropped_frame, model, device, transform)
            
            column = cropped_mask[:, x_expected_value]
            diameter = np.sum(column) / 255
            if diameter > 0:
                diameters.append(diameter * w_ratio)  # Convert to original scale
                top_point = np.where(column == 255)[0][0]
            else:
                top_point = None


            cropped_frame = cv.resize(cropped_frame, (512, 512))
            if top_point:
                cv.line(cropped_frame, (x_expected_value, top_point), (x_expected_value, top_point + int(diameter)), (0, 255, 0), 2)

            
            cropped_mask = cv.cvtColor(cropped_mask, cv.COLOR_GRAY2BGR)
            combined_cropped = np.hstack((cropped_frame, cropped_mask))

            cv.imshow('Cropped Frame | Cropped Mask', combined_cropped)

            key_code = cv.waitKey(1)
            if key_code == 27:  # ESC key to exit
                break

            if fluid_broke or current_frame >= total_frames - 1:
                break
        
        # Create scatter plot of diameters
        if diameters:
            
            diameters = np.array(diameters)
            diameters = diameters / diameters[0]

            plt.figure(figsize=(10, 6))
            plt.scatter(np.arange(len(diameters)), diameters)
            plt.xlabel('Frame Index')
            plt.ylabel('Diameter')
            plt.title('Diameter Over Time')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'out/{video}.png')


    
if __name__ == "__main__":
    main()