import rclpy
from rclpy.node import Node

def main(node: Node): 

    node.get_logger().info(f"Robot pose -> x:{node.robot_pose.x} | y: {node.robot_pose.y } | theta: {node.robot_pose.theta}")
   # node.get_logger().info(f"Moviment direction -> {node.movement_direction}")
    node.get_logger().info(f"Error -> linear: {node.error_linear} | angular: {node.error_angle}")
    node.get_logger().info(f"Velocity -> publish: {node.robot_state == node.ROBOT_AUTONOMOUS} | linear: {node.cmd_vel.linear.x} | angular: {node.cmd_vel.angular.z}\n")
