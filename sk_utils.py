import os
import sys
import random
import yaml

import numpy as np
import cv2
import torch
import numexpr as ne

class Utils:

    def check_gpus():

        print(f"GPUs present?\t{torch.cuda.is_available()}\t{torch.cuda.device_count()}")
        for d in range(torch.cuda.device_count()):
            print(f"Device {d}: {torch.cuda.get_device_name(d)}")

    def read_params(path:str="params.yaml"):
        """read parameters from yaml file for RGC details

        Args:
            path (str, optional): path to yaml file. Defaults to "params.yaml".
        """

        # set the default loader to handle math expressions
        def expr_contructor(loader, node):
            expr = loader.construct_scalar(node)
            return float(ne.evaluate(expr).item())
        yaml.SafeLoader.add_constructor("!expr", expr_contructor)

        with open(path, 'r') as file:
            params = yaml.safe_load(file)

        # convert params to int/float if necessary (just in case they havent been read properly)
        for key, value in params.items():
            if isinstance(value, str):
                try:
                    params[key] = float(value)
                except ValueError:
                    pass

        return params
    
    def read_video(path:str):

        # open the video and get properties
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # read the actual frames in B&W
        frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.)
        cap.release()

        # return the frames and relevant properties
        return np.array(frames), fps
