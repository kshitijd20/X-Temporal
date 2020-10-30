import cv2
import os
import glob
import argparse



def main():
    parser = argparse.ArgumentParser(description='ROIs from sabine kastners atlas vs. tasknomy grouped RDMs')
    parser.add_argument('--video_dir', help='video_dir', default = 'D:/Projects/Algonauts2020/AlgonautsVideos268_All_30fpsmax', type=str)
    args = vars(parser.parse_args())
    video_dir = args['video_dir']
    
    video_list = glob.glob(video_dir+'/*.mp4')
    video_list.sort()
    print(len(video_list))
    with open("Output.txt", "w") as text_file:
        for v_count,video_file in enumerate(video_list):
            #print(video_file)
            cap = cv2.VideoCapture(video_file)
            num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            #print(int(num_frames))
            text_file.write(video_file+" "+str(int(num_frames))+" "+ str(0) + "\n")


if __name__ == "__main__":
    main()