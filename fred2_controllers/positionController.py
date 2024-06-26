#!/user/bin/env python3

import rclpy
import transforms3d as tf3d     # angle manipulaton 
import threading 
import math
import sys

import subscribers

from typing import List

from fred2_controllers.lib.PID import PID_controller

from fred2_controllers.lib.quat_multiply import quaternion_multiply, reduce_angle

from rclpy.context import Context

from rclpy.node import Node, ParameterDescriptor
from rclpy.parameter import Parameter, ParameterType
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSPresetProfiles, QoSProfile, QoSHistoryPolicy, QoSLivelinessPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

from rcl_interfaces.msg import SetParametersResult
from rcl_interfaces.srv import GetParameters

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D, PoseStamped, Pose, Quaternion, Twist
from std_msgs.msg import Int16 

# args 
debug_mode = "--debug" in sys.argv


class positionController (Node): 
    
    odom_pose = Pose()                          # Current position of the robot obtained from odometry

    goal_pose = Pose2D()                        # Goal position for the robot to navigate towards
    robot_state = -1                            # Current state of the robot, starts in random value

    movement_direction = 1                      # Direction of movement: 1 for forward, -1 for backward  

    robot_quat = Quaternion()                   # Quaternion representing the orientation of the robot  
    robot_pose = Pose2D()                       # Pose of the robot (position and orientation)

    bkward_pose = Pose2D()                      # Pose when moving backward
    front_pose = Pose2D()                       # Pose when moving forward

    cmd_vel = Twist()                           # Twist message for velocity commands

    rotation_quat = [0.0, 0.0, 0.0, 0.0]        # Quaternion used for rotation
    robot_quat = [0.0, 0.0, 0.0, 0.0]           # Quaternion representing the orientation of the robot


    # starts with randon value 
    ROBOT_MANUAL = 1000
    ROBOT_AUTONOMOUS = 1000
    ROBOT_IN_GOAL = 1000
    ROBOT_MISSION_COMPLETED = 1000
    ROBOT_EMERGENCY = 1000



    def __init__(self, 
                node_name: str, 
                *, # keyword-only argument
                context: Context = None, 
                cli_args: List[str] = None, 
                namespace: str = None, 
                use_global_arguments: bool = True, 
                enable_rosout: bool = True, 
                start_parameter_services: bool = True, 
                parameter_overrides: List[Parameter] | None = None) -> None:
        
        super().__init__(node_name=node_name, 
                        context=context, 
                        cli_args=cli_args, 
                        namespace=namespace, 
                        use_global_arguments=use_global_arguments, 
                        enable_rosout=enable_rosout, 
                        start_parameter_services=start_parameter_services, 
                        parameter_overrides=parameter_overrides)
        
        self.quality_protocol()
        self.setup_subscribers()
        self.setup_publishers()

        self.load_params()
        self.get_params()

        self.add_on_set_parameters_callback(self.parameters_callback)    



    def quality_protocol(self):

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,  # Set the reliability policy to RELIABLE, ensuring reliable message delivery
            durability= QoSDurabilityPolicy.VOLATILE,   # Set the durability policy to VOLATILE, indicating messages are not stored persistently
            history=QoSHistoryPolicy.KEEP_LAST,         # Set the history policy to KEEP_LAST, storing a limited number of past messages
            depth=10,                                   # Set the depth of the history buffer to 10, specifying the number of stored past messages
            liveliness=QoSLivelinessPolicy.AUTOMATIC    # Set the liveliness policy to AUTOMATIC, allowing automatic management of liveliness 
    )
        
    def setup_publishers(self): 

        # ----- Publish velocity command for reaches the goal 
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 5)
        

    # Declare params from the yaml file 
    def load_params(self): 
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('kp_angular', None, 
                    ParameterDescriptor(
                        description='Proportional gain for angular movement', 
                        type=ParameterType.PARAMETER_DOUBLE)),

                ('ki_angular', None, 
                    ParameterDescriptor(
                        description='Integrative gain for angular movement', 
                        type=ParameterType.PARAMETER_DOUBLE)),

                ('kd_angular', None, 
                    ParameterDescriptor(
                        description='Derivative gain for angular movement', 
                        type=ParameterType.PARAMETER_DOUBLE)),

                ('kp_linear', None, 
                    ParameterDescriptor(
                        description='Proportional gain for linear movement', 
                        type=ParameterType.PARAMETER_DOUBLE)),

                ('ki_linear', None, 
                    ParameterDescriptor(
                        description='Integrative gain for linear movement', 
                        type=ParameterType.PARAMETER_DOUBLE)),

                ('kd_linear', None, 
                    ParameterDescriptor(
                        description='Derivative gain for linear movement', 
                        type=ParameterType.PARAMETER_DOUBLE)),

                ('max_linear_vel', None, 
                    ParameterDescriptor(
                        description='Max linear velocity in a straight line', 
                        type=ParameterType.PARAMETER_DOUBLE)),

                ('min_linear_vel', None, 
                    ParameterDescriptor(
                        description='Min linear speed for rotational movement', 
                        type=ParameterType.PARAMETER_DOUBLE)),

                ('debug', None, 
                    ParameterDescriptor(
                        description='Enable debug prints', 
                        type=ParameterType.PARAMETER_BOOL)), 

                ('frequency', None, 
                    ParameterDescriptor(
                        description='Node frequency', 
                        type=ParameterType.PARAMETER_INTEGER)),

                ('unit_test', None, 
                    ParameterDescriptor(
                        description='Allows the node to run isolated', 
                        type=ParameterType.PARAMETER_BOOL)),
            ]
        )

        self.get_logger().info('All parameters successfully declared')


    # updates the parameters when they are changed by the command line
    def parameters_callback(self, params):  
        
        for param in params:
            self.get_logger().info(f"Parameter '{param.name}' changed to: {param.value}")


        if param.name == 'kp_angular':
            self.KP_ANGULAR = param.value
    
  
        if param.name == 'kd_angular':
            self.KD_ANGULAR = param.value


        if param.name == 'ki_angular': 
            self.KI_ANGULAR = param.value


        if param.name == 'max_linear_vel': 
            self.MAX_LINEAR_VEL = param.value
        

        if param.name == 'min_linear_vel': 
            self.MIN_LINEAR_VEL = param.value

        
        if param.name == 'debug': 
            self.DEBUG = param.value
        

        if param.name == 'unit_test': 
            self.UNIT_TEST = param.value
        

        if param.name == 'frequency': 
            self.FREQUENCY = param.value 


        return SetParametersResult(successful=True)


    # get the param value from the yaml file
    def get_params(self): 
        
        self.KP_ANGULAR = self.get_parameter('kp_angular').value
        self.KI_ANGULAR = self.get_parameter('ki_angular').value
        self.KD_ANGULAR = self.get_parameter('kd_angular').value

        self.MAX_LINEAR_VEL = self.get_parameter('max_linear_vel').value
        self.MIN_LINEAR_VEL = self.get_parameter('min_linear_vel').value

        self.DEBUG = self.get_parameter('debug').value
        self.UNIT_TEST = self.get_parameter('unit_test').value
        self.FREQUENCY = self.get_parameter('frequency').value



        # if the unit test is active, it disabled the global param from machine states 
        if self.UNIT_TEST: 
            
            self.robot_state = 2
            self.get_logger().info('In UNIT TEST mode')  

        
        else: 

            # Get global params 
            self.client = self.create_client(GetParameters, '/machine_states/main_robot/get_parameters')
            self.client.wait_for_service()

            request = GetParameters.Request()
            request.names = ['manual', 'autonomous', 'in_goal', 'mission_completed', 'emergency']

            future = self.client.call_async(request)
            future.add_done_callback(self.callback_global_param)
        



    # get the global values from the machine states params 
    def callback_global_param(self, future):


        try:

            result = future.result()

            self.ROBOT_MANUAL = result.values[0].integer_value
            self.ROBOT_AUTONOMOUS = result.values[1].integer_value
            self.ROBOT_IN_GOAL = result.values[2].integer_value
            self.ROBOT_MISSION_COMPLETED = result.values[3].integer_value
            self.ROBOT_EMERGENCY = result.values[4].integer_value


            self.get_logger().info(f"Got global param ROBOT_MANUAL -> {self.ROBOT_MANUAL}")
            self.get_logger().info(f"Got global param ROBOT_AUTONOMOUS -> {self.ROBOT_AUTONOMOUS}")
            self.get_logger().info(f"Got global param ROBOT_IN GOAL -> {self.ROBOT_IN_GOAL}")
            self.get_logger().info(f"Got global param ROBOT_MISSION_COMPLETED: {self.ROBOT_MISSION_COMPLETED}")
            self.get_logger().info(f"Got global param ROBOT_EMERGENCY: {self.ROBOT_EMERGENCY}\n")



        except Exception as e:

            self.get_logger().warn("Service call failed %r" % (e,))



        

    
    
    def move_backward(self): 

        # Convert 180-degree rotation to quaternion
        rotation_180_degree_to_quat = tf3d.euler.euler2quat(0, 0, math.pi)   # Quaternion in w, x, y z (real, then vector) format
        
        # Assign quaternion components for rotation
        self.rotation_quat[0] = rotation_180_degree_to_quat[1]
        self.rotation_quat[1] = rotation_180_degree_to_quat[2]
        self.rotation_quat[2] = rotation_180_degree_to_quat[3]
        self.rotation_quat[3] = rotation_180_degree_to_quat[0]

        # Get current orientation of the robot
        self.robot_quat[0] = self.odom_pose.orientation.x 
        self.robot_quat[1] = self.odom_pose.orientation.y 
        self.robot_quat[2] = self.odom_pose.orientation.z
        self.robot_quat[3] = self.odom_pose.orientation.w 

        # Calculate the quaternion for the backward movement
        backwart_quat = []
        backwart_quat = quaternion_multiply(self.robot_quat, self.rotation_quat)


        backwart_pose = Pose2D()
        backwart_pose.x = self.odom_pose.position.x 
        backwart_pose.y = self.odom_pose.position.y 

        # Calculate theta for the backward pose
        backwart_pose.theta = tf3d.euler.quat2euler([backwart_quat[3], 
                                                    backwart_quat[0], 
                                                    backwart_quat[1], 
                                                    backwart_quat[2]])[2]


        return backwart_pose
    



    def move_front(self): 
        
        # Get current orientation quaternion of the robot
        front_quat = Quaternion()
        front_quat = self.odom_pose.orientation

        front_pose = Pose2D()
        front_pose.x = self.odom_pose.position.x 
        front_pose.y = self.odom_pose.position.y 

        # Calculate theta for the front pose
        front_pose.theta = tf3d.euler.quat2euler([front_quat.w, 
                                                front_quat.x, 
                                                front_quat.y, 
                                                front_quat.z])[2]
        

        return front_pose




    def position_control (self): 
        
        # Determine the direction of movement based on the movement_direction variable
        if self.movement_direction == 1: 
            
            self.robot_pose = self.move_front()

        elif self.movement_direction == -1: 
            
            self.robot_pose = self.move_backward()
        

        # Calculate the error between the goal pose and the robot pose
        dx = self.goal_pose.x - self.robot_pose.x 
        dy = self.goal_pose.y - self.robot_pose.y 

        error_linear = math.hypot(dx, dy)
        error_angle = math.atan2(dy, dx)

        # Calculate heading errors for backward movement
        self.bkward_pose = self.move_backward()
        bkward_heading_error = reduce_angle(error_angle - self.bkward_pose.theta)


        # Calculate heading errors for forward movement
        self.front_pose = self.move_front()
        front_heading_error = reduce_angle(error_angle - self.front_pose.theta)


        # Switch movement direction if necessary to minimize heading error
        if (abs(front_heading_error) > abs(bkward_heading_error) and (self.movement_direction == 1)):
            
            self.movement_direction = -1 
            self.robot_pose = self.move_backward()

            self.get_logger().warn('Switching to backwards orientation')


        # Switch movement direction if necessary to minimize heading error
        if (abs(front_heading_error) < abs(bkward_heading_error) and (self.movement_direction == -1)): 
            
            self.movement_direction = 1 
            self.robot_pose = self.move_front()

            self.get_logger().warn('Switching to foward orientation')


        # Calculate orientation error
        orientation_error = reduce_angle(error_angle - self.robot_pose.theta)


        # Calculate angular velocity using PID controller
        angular_vel = PID_controller(self.KP_ANGULAR, self.KI_ANGULAR, self.KD_ANGULAR)


        # Calculate linear velocity based on orientation error
        if error_linear != 0:

            self.cmd_vel.linear.x = ((1-abs(orientation_error)/math.pi)*(self.MAX_LINEAR_VEL - self.MIN_LINEAR_VEL) + self.MIN_LINEAR_VEL) * self.movement_direction
        
        else: 

            self.cmd_vel.linear.x = 0.0


        # Set angular velocity
        self.cmd_vel.angular.z = angular_vel.output(orientation_error)


        # Publish velocity if the robot is in autonomous mode
        if self.robot_state == self.ROBOT_AUTONOMOUS: 
            
            self.vel_pub.publish(self.cmd_vel)


        if debug_mode or self.DEBUG:

            self.get_logger().info(f"Robot pose -> x:{self.robot_pose.x} | y: {self.robot_pose.y } | theta: {self.robot_pose.theta}")
            self.get_logger().info(f"Moviment direction -> {self.movement_direction}")
            self.get_logger().info(f"Error -> linear: {error_linear} | angular: {error_angle}")
            self.get_logger().info(f"Velocity -> publish: {self.robot_state == self.ROBOT_AUTONOMOUS} | linear: {self.cmd_vel.linear.x} | angular: {self.cmd_vel.angular.z}\n")



def main(): 
    
    # Create a custom context for single thread and real-time execution
    rclpy.init()

    position_context = rclpy.Context()
    position_context.init()
    position_context.use_real_time = True
    
    node = positionController(
        node_name='positionController',
        context=position_context,
        cli_args=['--debug'],
        namespace='controllers',
        enable_rosout=False
    )

    # Make the execution in real-time 
    executor = SingleThreadedExecutor(context=position_context)
    executor.add_node(node)

    # Create a separate thread for the callbacks and another for the main function 
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    rate = node.create_rate(node.FREQUENCY)

    try: 
        while rclpy.ok(): 
            rate.sleep()
            node.position_control()
        
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    node.destroy_node()
    thread.join()




if __name__ == '__main__':
    
    main()
