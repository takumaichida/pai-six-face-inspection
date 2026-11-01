"""ROS 2 node: the conveyor line (MuJoCo) and the robot program.

When the program takes a picture it publishes it on /inspection/image and waits for the inspector's verdict
on /inspection/verdict, so judging runs in a separate node (and could run on another machine).

Publishes   /inspection/image  (sensor_msgs/Image)
            /joint_states      (sensor_msgs/JointState, the 7 arm joints + finger)
            /line/status       (std_msgs/String, what the robot is doing)
            /line/result       (std_msgs/String, JSON per part: decision, views, cycle time)
Subscribes  /inspection/verdict (std_msgs/String, JSON)
"""
import json
import os
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from sortbot.lane import N_PARTS, LaneCell  # noqa: E402
from sortbot.lane_routine import LaneProgram  # noqa: E402


class LineNode(Node):
    def __init__(self):
        super().__init__("conveyor_line")
        self.declare_parameter("seed", 3)
        self.declare_parameter("defects", 3)
        self.pub_img = self.create_publisher(Image, "/inspection/image", 10)
        self.pub_js = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_status = self.create_publisher(String, "/line/status", 10)
        self.pub_result = self.create_publisher(String, "/line/result", 10)
        self.create_subscription(String, "/inspection/verdict", self.on_verdict, 10)
        self.verdicts, self.seq = {}, 0
        self.cell = LaneCell(video_size=(320, 180))
        self.cell.reset(seed=self.get_parameter("seed").value, n_defect=self.get_parameter("defects").value)
        self.prog = LaneProgram(self.cell, self.remote_judge)
        self.cell.frame_hook = self.on_step
        self.last_status = ""
        self.joint_names = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1"]

    def on_verdict(self, msg):
        v = json.loads(msg.data)
        self.verdicts[v["seq"]] = v

    def remote_judge(self, cam, img, part):
        """Send the face image to the inspector node and wait for its verdict."""
        self.seq += 1
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"{cam}|{part}|{self.seq}"
        msg.height, msg.width = img.shape[:2]
        msg.encoding, msg.step = "rgb8", img.shape[1] * 3
        msg.data = img.tobytes()
        self.pub_img.publish(msg)
        t0 = time.time()
        while self.seq not in self.verdicts:
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() - t0 > 10:
                raise RuntimeError("no verdict from the inspector node")
        return self.verdicts.pop(self.seq)["p_defect"]

    def on_step(self, cell):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.joint_names
        d = cell.data
        js.position = [float(x) for x in d.qpos[cell.arm_qadr]] + [float(d.qpos[cell.finger_qadr])]
        self.pub_js.publish(js)
        if self.prog.status != self.last_status:
            self.last_status = self.prog.status
            self.pub_status.publish(String(data=self.prog.status))
            self.get_logger().info(self.prog.status)

    def run(self):
        while self.pub_img.get_subscription_count() == 0:     # wait for the inspector
            rclpy.spin_once(self, timeout_sec=0.2)
        for i in range(N_PARTS):
            r = self.prog.run_part(i)
            r = dict(part=i + 1, decision=r["decision"], views=r["n_views"],
                     cycle_s=round(r["t_done"] - r["t_pick"], 1), defects=len(self.cell.defect_list[i]))
            self.pub_result.publish(String(data=json.dumps(r)))
            self.get_logger().info(f"RESULT {r}")


def main():
    rclpy.init()
    node = LineNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
