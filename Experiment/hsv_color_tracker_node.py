import rospy
import numpy as np
from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image



class HSVColorTrackerNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.camera_topic = rospy.get_param('/color_tracker/image_topic')
        self.lower_hue = rospy.get_param('/color_tracker/lower_h')
        self.upper_hue = rospy.get_param('/color_tracker/upper_h')
        self.lower_saturation = rospy.get_param('/color_tracker/lower_s')
        self.upper_saturation = rospy.get_param('/color_tracker/upper_s')
        self.lower_value = rospy.get_param('/color_tracker/lower_v')
        self.upper_value = rospy.get_param('/color_tracker/upper_v')

        self.fx = rospy.get_param('/color_tracker/fx')
        self.fy = rospy.get_param('/color_tracker/fy')
        self.origin_x = rospy.get_param('/color_tracker/origin_x')
        self.origin_y = rospy.get_param('/color_tracker/origin_y')

        self.color_ranges = {
            'red': ([self.lower_hue, self.lower_saturation, self.lower_value],
                    [self.upper_hue, self.upper_saturation, self.upper_value])}
        self.trajectories = {name: [] for name in self.color_ranges.keys()}
        self.draw_colors = {'red': (0, 0, 255),
                            'black': (0, 0, 0),}

        self.image_subscriber = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        self.position_publisher = rospy.Publisher('hsv_tracker/position', PointStamped, queue_size=1)
        self.image_publisher = rospy.Publisher('hsv_tracker/image_out', Image, queue_size=1)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")
            return
        output, x_m, y_m = self.detect_and_draw_boxes(frame)

        if output is not None:
            self.publish_point(x_m, y_m)
            image_out = self.bridge.cv2_to_imgmsg(output, encoding='bgr8')
            self.image_publisher.publish(image_out)

        
        
    def calculate_position_meter(self, x, y):
        # Calculate the position in milimeters
        x_m = (x - self.origin_x) * self.fx
        y_m = (y - self.origin_y) * self.fy
        return x_m, y_m

    def detect_and_draw_boxes(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        output = frame.copy()
        
        for name, (lower, upper) in self.color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)

            # Clean mask with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter and merge contours
            filtered_contours = []
            for cnt in contours:
                if cv2.contourArea(cnt) < 30:  # Skip small noise
                    continue
                filtered_contours.append(cnt)

            # Merge close contours
            merged_contours = []
            while filtered_contours:
                base = filtered_contours.pop(0)
                x, y, w, h = cv2.boundingRect(base)
                merged = [base]
                remove_indices = []
                for idx, other in enumerate(filtered_contours):
                    ox, oy, ow, oh = cv2.boundingRect(other)
                    if abs(x - ox) < 20 and abs(y - oy) < 20:  # Threshold for merging
                        merged.append(other)
                        remove_indices.append(idx)
                for idx in sorted(remove_indices, reverse=True):
                    filtered_contours.pop(idx)
                merged_contours.append(cv2.boundingRect(np.vstack(merged)))
            if merged_contours:
                largest_box = max(merged_contours, key=lambda b: b[2] * b[3])  # Select by area
            else:
                rospy.logwarn(f"No contours found, no new position published waiting for the next frame")
                return None, 0, 0
            x, y, w, h = largest_box

            # Calculate the middle point of the bounding box
            middle_x = x + w // 2
            middle_y = y + h // 2
            self.trajectories[name].append((middle_x, middle_y))

            # Convert the middle point to meters
            x_m, y_m = self.calculate_position_meter(middle_x, middle_y)

            # Draw the bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), self.draw_colors[name], 2)

            # Plot the middle point and its position in meters
            cv2.circle(output, (middle_x, middle_y), 5, self.draw_colors[name], -1)

        return output, x_m, y_m
    
    def publish_point(self, x_m, y_m):
        position = PointStamped()
        position.header.stamp = rospy.Time.now()
        position.header.frame_id = 'mns'
        position.point.x = x_m
        position.point.y = y_m
        position.point.z = 0.0
        self.position_publisher.publish(position)

if __name__ == '__main__':
    rospy.init_node('color_tracker')

    node = HSVColorTrackerNode()

    rospy.spin()