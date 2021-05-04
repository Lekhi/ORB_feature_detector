import argparse
import sys
import cv2
import numpy as np
import inspect
import matplotlib.pyplot as plt


#constants
IMG_SCALE_PERCENT = 20 
VIDEO_SCALE_PERCENT = 80


def show_img(img, bw=False):
    """
    Showing image that will be example to use in object detecting.

    """
    fig = plt.figure(figsize=(13, 13))
    ax = fig.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(img, cmap='Greys_r' if bw else None)
    plt.show()


def print_interesting_members(obj):
    """
    Showing information about one of the matches.
    """
    for name, value in inspect.getmembers(obj):
        try:
            float(value)
            print(f'{name} -> {value}')
        except Exception:
            pass


def main(args):
    

    # preparing sample image 
    img = cv2.imread(args.picture)
    width = int(img.shape[1] * IMG_SCALE_PERCENT / 100)
    height = int(img.shape[0] * IMG_SCALE_PERCENT / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


    video = cv2.VideoCapture(args.input_video)

    if video is None:
        print(f'Unable to open {args.input_video}')
        sys.exit()
    
    if img is None:
        print(f'Unable to open {args.picture}')
        sys.exit()
    
    show_img(img)
    gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    show_img(gray_picture, True)

    # creating ORB feature detector
    orb = cv2.ORB_create()
    kp_img, des_img = orb.detectAndCompute(gray_picture, None)
    print(f'Keypoints in image: {len(kp_img)}')
    print(f'Example of keypoints:')
    print_interesting_members(kp_img[0])
    print(f'Example descriptor {des_img[0]}')

    keypoints_vis = cv2.drawKeypoints(img, kp_img, None)
    show_img(keypoints_vis)

    # creating matcher 
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    
    # reading first frame of video and detecting features
    frame_read, frame = video.read()
    width = int(frame.shape[1] * VIDEO_SCALE_PERCENT / 100)
    height = int(frame.shape[0] * VIDEO_SCALE_PERCENT / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)
    

    while frame_read:
        # main loop that iterates frame by frame whole video

        #finding matches
        matches = matcher.match(des_img, des_frame)
        print(f'Number of matches: {len(matches)}')
        print(f'Example of match:')
        print_interesting_members(matches[0])

        # chosing only best matches to avoid outliers
        # best == shortest distance
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:15]

        # calculating average distance of best matches to check their quality
        avg_distance = sum(match.distance for match in good_matches )/len(good_matches)
        
        # checking quality of matches by calculating average distance
        if avg_distance < 55:
            

            # extracting points from dMatch then using ransac to reject outliers
            # also transforming pespective to get proper angle of box around object
            src_pts = np.float32([ kp_img[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_frame[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            
            h, w, _ = img.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
         
            
            # draw a polyline around the found object:
            image = cv2.polylines(frame,[np.int32(dst)], 
                                isClosed = True,
                                color = (0,255,0),
                                thickness = 3, 
                                )  
            cv2.imshow('ORB detection demonstration', image)
           
        else:

            cv2.imshow('ORB detection demonstration', frame)
        key = cv2.waitKey(1)

        #it is possible to end script by pressing 'q' and it is also possible to pause video by pressing 'p'
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)
        
        # reading next frame, resizing it for easier work and converting color of frame to gray scale
        frame_read, frame = video.read()
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # using orb detector to find keypoints
        kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)



def parse_args():
    """
    Makes sure that required arguments has been provided.
    Script needs image and video to work.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_video', required=True)
    parser.add_argument('-p', '--picture', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())