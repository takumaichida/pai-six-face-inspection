#!/bin/bash
# Run the line and the AI inspector as two ROS 2 nodes and log the topics.
# usage: ros2/run_demo.sh [classifier|anomaly]   (needs a ROS 2 Humble environment; see README)
cd "$(dirname "$0")/.."
MODEL=${1:-classifier}
mkdir -p out
python ros2/inspector_node.py --ros-args -p model:=$MODEL > out/ros_inspector.log 2>&1 &
INSP=$!
ros2 topic echo /line/result std_msgs/msg/String > out/ros_results.log 2>&1 &
ECHO=$!
sleep 4
python ros2/line_node.py --ros-args -p seed:=3 -p defects:=3 > out/ros_line.log 2>&1 &
LINE=$!
sleep 15
ros2 node list > out/ros_nodes.txt
ros2 topic list -t > out/ros_topics.txt
wait $LINE
grep RESULT out/ros_line.log
kill -INT $INSP $ECHO
