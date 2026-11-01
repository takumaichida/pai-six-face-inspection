"""ROS 2 node: the AI inspector.

Subscribes  /inspection/image    (sensor_msgs/Image, header.frame_id = "<camera>|<part>|<seq>")
Publishes   /inspection/verdict  (std_msgs/String, JSON: seq, part, camera, p_defect, label, model)

Parameter `model`: "classifier" (ResNet-18 trained on ok / defect images) or
"anomaly" (good-only detector, no defect images in training).
"""
import json
import os
import sys

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
LABELS = ["ok", "scratch", "stain", "chip"]


class Inspector(Node):
    def __init__(self):
        super().__init__("inspector")
        self.declare_parameter("model", "classifier")
        kind = self.get_parameter("model").value
        if kind == "anomaly":
            from sortbot.anomaly import AnomalyJudge
            self.judge = AnomalyJudge(os.path.join(ROOT, "models/anomaly.pt"))
        else:
            from sortbot.judge import FaceJudge
            self.judge = FaceJudge(os.path.join(ROOT, "models/face_cls.pt"))
        self.kind = kind
        self.pub = self.create_publisher(String, "/inspection/verdict", 10)
        self.create_subscription(Image, "/inspection/image", self.on_image, 10)
        self.get_logger().info(f"inspector ready (model={kind})")

    def on_image(self, msg):
        cam, part, seq = msg.header.frame_id.split("|")
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        p = float(self.judge(cam, img, int(part)))
        probs = self.judge.last["probs"]
        label = LABELS[int(np.argmax(probs[1:]) + 1)] if p > 0.5 else "ok"
        out = dict(seq=int(seq), part=int(part), camera=cam, p_defect=round(p, 4), label=label, model=self.kind)
        self.pub.publish(String(data=json.dumps(out)))
        self.get_logger().info(f"part {int(part) + 1} {cam}: {label} (p={p:.2f})")


def main():
    rclpy.init()
    node = Inspector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
