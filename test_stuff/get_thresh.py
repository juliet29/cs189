import cv2
import rospy
import numpy as np

# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError


def nothing(x):
    """
    This is a function that does nothing. Useful for event-driven programs where
    you don't want the event to trigger an action.
    :param x: Anything
    :return: None
    """
    pass


class ViewImages:
    """
    Use the RGB camera to find the largest blue object
    """

    def __init__(self):
        # rospy.init_node('ViewImages', anonymous=False)

        # Set thresholds for color detection and thresholding
        self.thresh_low = np.array([3, 170, 128])
        self.thresh_high = np.array([17, 190, 208])

        # ctrl + c -> call self.shutdown function
        rospy.on_shutdown(self.shutdown)

        # # print msg
        # rospy.loginfo("Hello World!")

        # # How often should provide commands? 5 Hz
        # self.rate = rospy.Rate(5)

        # # Subscribe to topic for BGR images
        # rospy.Subscriber('/camera/rgb/image_raw', Image, self.process_image, queue_size=1, buff_size=2 ** 24)
        # self.bridge = CvBridge()

    def run(self):
        """
        Run the robot until Ctrl+C
        We're not moving, so not much happens here
        :return: None
        """
        
        # Create a black image, a window
        img_width = 512
        img = np.zeros((300, img_width, 3), np.uint8)
        cv2.namedWindow('thresholds')

        # Create trackbars for color change
        cv2.createTrackbar('H (low)', 'thresholds', 0, 179, nothing)
        cv2.createTrackbar('S (low)', 'thresholds', 0, 255, nothing)
        cv2.createTrackbar('V (low)', 'thresholds', 0, 255, nothing)
        cv2.createTrackbar('H (high)', 'thresholds', 0, 179, nothing)
        cv2.createTrackbar('S (high)', 'thresholds', 0, 255, nothing)
        cv2.createTrackbar('V (high)', 'thresholds', 0, 255, nothing)
        
        # Set with saved/inial values
        cv2.setTrackbarPos('H (low)', 'thresholds', self.thresh_low[0])
        cv2.setTrackbarPos('S (low)', 'thresholds', self.thresh_low[1])
        cv2.setTrackbarPos('V (low)', 'thresholds', self.thresh_low[2])
        cv2.setTrackbarPos('H (high)', 'thresholds', self.thresh_high[0])
        cv2.setTrackbarPos('S (high)', 'thresholds', self.thresh_high[1])
        cv2.setTrackbarPos('V (high)', 'thresholds', self.thresh_high[2])

        while(True):
            cv2.imshow('thresholds', img)

            # Get current positions of trackbars
            h = cv2.getTrackbarPos('H (low)', 'thresholds')
            s = cv2.getTrackbarPos('S (low)', 'thresholds')
            v = cv2.getTrackbarPos('V (low)', 'thresholds')
            img[:, :img_width/2] = [h, s, v]
            self.thresh_low = np.array([h, s, v])
            
            h = cv2.getTrackbarPos('H (high)', 'thresholds')
            s = cv2.getTrackbarPos('S (high)', 'thresholds')
            v = cv2.getTrackbarPos('V (high)', 'thresholds')
            img[:, img_width/2:] = [h, s, v]
            self.thresh_high = np.array([h, s, v])
            
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
                
            # self.rate.sleep()

    def threshold(self, img):
        """
        Apply the threshold to the image to get only blue parts
        :param img: BGR camera image
        :return: Masked image
        """

        # Apply a blur
        img = cv2.medianBlur(img, 5)
        # Convert image to easier-to-work-with HSV format
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Threshold to get only blue values
        mask = cv2.inRange(img_hsv, self.thresh_low, self.thresh_high)
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        return masked_img

    def process_image(self, data):
        """
        Process an BGR image from the camera whenever received
        :param data: Raw BGR image
        :return: None
        """
        try:
            cv2.imshow('picture', cv_image)
            cv2.waitKey(3)
            # Convert the image from ROS format to OpenCV format
            # 'bgr8' means it will encode as 8-bit values in BGR channels
            #cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # Apply a threshold to your image
            #cv_image = self.threshold(cv_image)
            # Display the modified imag



if __name__ == '__main__':
    try:
        robot = ViewImages()
        robot.run()
    except Exception, err:
        rospy.loginfo("ViewImages node terminated.")
        print err
