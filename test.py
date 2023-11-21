import argparse
import torch
import cv2 as cv

from src.tetris import Tetirs

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris"""
    )

    parser.add_argument("--width", default=10, type=int)
    parser.add_argument("--height", default=20, type=int)
    parser.add_argument("--block_size", default=30, type=int)
    parser.add_argument("--fps", default=300, type=int)
    parser.add_argument("--saved_path", default="trained_model", type=str)
    parser.add_argument("--output", default="output.mp4", type=str)

    args = parser.parse_args()

    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if torch.cuda.is_available():
        model = torch.load(f'{opt.save_path}/tetris')
    else:
        model = torch.load(f'{opt.save_path}/tetris', map_location=lambda storage, loc: storage)
    
    model.eval()
    
    env = Tetirs(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()

    if torch.cuda.is_available():
        model.cuda()

    out = cv.VideoWriter(opt.output, cv.VideoWriter_fourcc(*"MJPG"), opt.fps,
                         (int(1.5 * opt.width * opt.block_size), opt.height * opt.block_size))

    while True:
        next_steps = env.get_next_state()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)

        if torch.cuda.is_available():
            next_states = next_states.cuda()
        
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]

        _, done = env.steps(action, render=True, video=out)

        if done:
            out.release()
            break


if __name__ == '__main__':
    opt = get_args()
    test(opt)
