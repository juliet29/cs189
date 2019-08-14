import numpy as np
import cv2 
import math
import datetime as dt 
# http://code.activestate.com/recipes/577231-discrete-pid-controller/ 

def centroid(contour):
    """
    Compute the (x,y) centroid position of the counter
    :param contour: OpenCV contour
    :return: Tuple of (x,y) centroid position
    """

    def centroid_x(c):
        """
        Get centroid x position
        :param c: OpenCV contour
        :return: x position or -1
        """
        M = cv2.moments(c)
        if M['m00'] == 0:
            return -1
        return int(M['m10'] / M['m00'])

    def centroid_y(c):
        """
        Get centroid y position
        :param c: OpenCV contour
        :return: y position or -1
        """
        M = cv2.moments(c)
        if M['m00'] == 0:
            return -1
        return int(M['m01'] / M['m00'])

    return centroid_x(contour), centroid_y(contour)

def dist(pos1, pos2):
    """
    Distance between 2 (x,y) positions
    """
    dpx = math.sqrt((pos1[0] - pos2[0]) ** 2 +
                     (pos1[1] - pos2[1]) ** 2)
    
    return dpx * 0.0167

    



class Follow:
    """
    follow a green using mac camera  
    """

    def __init__(self):
        self.name = 'Juliet'

        # only want to see green
        self.low_bound = np.array([23, 74, 0])
        self.high_bound = np.array([45, 255,  255])

        # parameters 
        self.band_timeout = dt.timedelta(seconds=10)

        # band info
        self.band_pos = None 
        self.band_seen_time = dt.datetime.now()



    def run(self):
        # for testing puposes
        print('hello ' + self.name)
            

    def process_image(self):
        # establish that video will come from mac camera 
        cap = cv2.VideoCapture(0)

        while(True):
            # capture frame by frame video as img
            # ret stores whether or not frame capture was succesful
            ret, img = cap.read() 

            # convert from rgb to hsv
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # apply blur to image 
            blur_img = cv2.medianBlur(hsv_img, 3)

            # threshold to only get orange values, apply mask that will show us only orange parts of img
            green_mask = cv2.inRange(blur_img, self.low_bound, self.high_bound)
            green_img = cv2.bitwise_and(img, img, mask=green_mask)

            # get contours 
            grayscale_img = cv2.cvtColor(green_img, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(grayscale_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                # find the biggest countour
                areas = [cv2.contourArea(c) for c in contours] # put all the areas of contours inside a list 
                max_index = np.argmax(areas) # find the biggest element in the list 
                max_contour = contours[max_index] # max contour at index of biggest element in list 
                
                # get the area and the centroid of this area
                new_band_pos = centroid(max_contour)
                new_band_area = areas[max_index]

                # draw contour and centroid on original image 
                cv2.drawContours(img, max_contour, -1, color=(0, 255, 0), thickness=2)
                cv2.circle(img, new_band_pos, radius=10, color=(255, 0, 0), thickness=2)
                

                # show the depth on an image
                cam_pos = (0,0)
                depth = round(dist(cam_pos, new_band_pos), 3)
                font1 = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(depth) + 'inches', (10, 500), font1, 2, color=(255, 0, 0))

                # check that the new_band is not too far from previous position 
                # (and that the area of the object is not too small)
                if (self.band_pos is None or dist(new_band_pos, self.band_pos) < 100) and new_band_area > 2000:
                    # update position and time seen 
                    self.band_pos = new_band_pos
                    self.band_seen_time = dt.datetime.now()

                # if the band has not been seen for a while -- the timeout time, then beep 
                if (dt.datetime.now() > self.band_seen_time + self.band_timeout):
                    self.band_pos = None
                    cv2.putText(img, 'lost band!!', (10, 100), font1, 2, color=(255, 0, 0))
                    # look for band 
                    # rotate until find band
                    # reset position to be moving in direction of band 

                
                # # keep the depth at 3 feet 
                # if (depth > 3 feet):
                #     cv2.putText(img, 'too far - adjusting', (10, 200), font1, 2, color=(255, 0, 0))
                #     # increase linear speed 

                # if (depth < 3 feet):
                #     cv2.putText(img, 'too close - adjusting', (10, 200), font1, 2, color=(255, 0, 0))
                #     # decrease linear speed 






            # show the image 
            cv2.imshow('img', img)

            # show the grayscale image
            #cv2.imshow('gray', grayscale_img)

            # show the mask
            #cv2.imshow('masked image', green_mask)

        
            # press q to quit image
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

       
                
            
    

if __name__ == '__main__':
    try:
        tester = Follow()
        tester.run()
        tester.process_image()
    except Exception, err:
        print err

