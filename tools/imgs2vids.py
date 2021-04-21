import argparse
import cv2
import os
import multiprocessing as mp
from functools import partial


def images2video(vid_dir, fps):
    vid_path = vid_dir + '.mp4'
    img_names = sorted(os.listdir(vid_dir))
    img_paths = [os.path.join(vid_dir, img_name) for img_name in img_names]

    height, width = cv2.imread(img_paths[0]).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(vid_path, fourcc, fps, (width, height))
    for img_path in img_paths:
        img = cv2.imread(img_path)
        writer.write(img)

    writer.release()
    print(f'{vid_dir} is finished.')


def main():
    parser = argparse.ArgumentParser(description='Tranfer images to videos')
    parser.add_argument('path', help='path to the output')
    parser.add_argument('--workers', type=int, default=4, help='thread number')
    parser.add_argument('--fps', type=int, default=5, help='video fps')
    args = parser.parse_args()

    base = args.path
    vid_dirs = [os.path.join(base, vid_name) for vid_name in os.listdir(base)]

    img2vid = partial(images2video, fps=args.fps)
    pool = mp.Pool(args.workers)
    pool.map(img2vid, vid_dirs)


if __name__ == '__main__':
    main()