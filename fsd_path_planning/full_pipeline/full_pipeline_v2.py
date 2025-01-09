#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: A class that runs the whole path planning pipeline.

- Cone sorting
- Cone Matching
- Path Calculation

Project: fsd_path_planning
"""
from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np

from fsd_path_planning.calculate_path.core_calculate_path import PathCalculationInput
from fsd_path_planning.cone_matching.core_cone_matching import ConeMatchingInput
from fsd_path_planning.config import (
    create_default_cone_matching_with_non_monotonic_matches,
    create_default_pathing,
    create_default_sorting,
)
from fsd_path_planning.relocalization.acceleration.acceleration_relocalization import (
    AccelerationRelocalizer,
)
from fsd_path_planning.relocalization.relocalization_base_class import Relocalizer
from fsd_path_planning.relocalization.relocalization_information import (
    RelocalizationInformation,
)
from fsd_path_planning.relocalization.skidpad.skidpad_path_data import BASE_SKIDPAD_PATH
from fsd_path_planning.relocalization.skidpad.skidpad_relocalizer import (
    SkidpadRelocalizer,
)
from fsd_path_planning.sorting_cones.core_cone_sorting import ConeSortingInput
from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import (
    angle_from_2d_vector,
    unit_2d_vector_from_angle,
)
from fsd_path_planning.utils.mission_types import MissionTypes
from fsd_path_planning.utils.utils import Timer

from yolo.msg import LabeledPointArray 
import rospy
import matplotlib.pyplot as plt
import geometry_msgs.msg
import std_msgs.msg
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import time  # 新增時間模組

import threading
import queue


MissionToRelocalizer: dict[MissionTypes, Relocalizer] = {
    MissionTypes.acceleration: AccelerationRelocalizer,
    MissionTypes.ebs_test: AccelerationRelocalizer,
    MissionTypes.skidpad: SkidpadRelocalizer,
}


class PathPlanner:
    def __init__(
        self, mission: MissionTypes, experimental_performance_improvements: bool = False
    ) -> None:
        self.mission = mission

        self.relocalizer: Relocalizer | None = None
        relocalizer_class = MissionToRelocalizer.get(mission)

        if relocalizer_class is not None:
            self.relocalizer = relocalizer_class()

        self.cone_sorting = create_default_sorting(
            mission, experimental_performance_improvements
        )

        self.cone_matching = create_default_cone_matching_with_non_monotonic_matches(
            mission
        )
        self.pathing = create_default_pathing(mission)
        self.global_path: Optional[FloatArray] = None

        self.experimental_performance_improvements = (
            experimental_performance_improvements
        )

    def _convert_direction_to_array(self, direction: Any) -> FloatArray:
        direction = np.squeeze(np.array(direction))
        if direction.shape == (2,):
            return direction

        if direction.shape in [(1,), ()]:
            return unit_2d_vector_from_angle(direction)

        raise ValueError("direction must be a float or a 2 element array")

    def set_global_path(self, global_path):
        self.global_path = global_path

    def calculate_path_in_global_frame(
        self,
        cones: List[FloatArray],
        vehicle_position: FloatArray,
        vehicle_direction: Union[FloatArray, float],
        return_intermediate_results: bool = False,
    ) -> Union[
        FloatArray,
        tuple[
            FloatArray,
            FloatArray,
            FloatArray,
            FloatArray,
            FloatArray,
            IntArray,
            IntArray,
        ],
    ]:
        rospy.loginfo(f"Initial vehicle position: {vehicle_position}")

        vehicle_direction = self._convert_direction_to_array(vehicle_direction)

        noprint = True

        if self.relocalizer is not None:
            with Timer("Relocalization", noprint=noprint):
                self.relocalizer.attempt_relocalization_calculation(
                    cones.astype(np.float64), 
                    vehicle_position.astype(np.float64), 
                    vehicle_direction.astype(np.float64)
                )

            if self.relocalizer.is_relocalized:
                print("重定位\n")
                vehicle_yaw = angle_from_2d_vector(vehicle_direction)
                (
                    vehicle_position,
                    vehicle_yaw,
                ) = self.relocalizer.transform_to_known_map_frame(
                    vehicle_position, vehicle_yaw
                )
                vehicle_direction = unit_2d_vector_from_angle(vehicle_yaw)
                self.global_path = self.relocalizer.get_known_global_path()

            
            sorted_left, sorted_right = np.zeros((2, 0, 2))
            left_cones_with_virtual, right_cones_with_virtual = np.zeros((2, 0, 2))
            left_to_right_match, right_to_left_match = np.zeros((2, 0), dtype=int)

        else:
            with Timer("Cone sorting", noprint=noprint):
                
                cone_sorting_input = ConeSortingInput(
                    cones.astype(np.float64), 
                    vehicle_position.astype(np.float64), 
                    vehicle_direction.astype(np.float64)
                )
                
                self.cone_sorting.set_new_input(cone_sorting_input)
                sorted_left, sorted_right = self.cone_sorting.run_cone_sorting()


                # 打印排序結果
                print(f"Sorted left cones: {sorted_left}") 
                print(f"Sorted right cones: {sorted_right}")


            with Timer("Cone matching", noprint=noprint):
                matched_cones_input = [np.zeros((0, 2)) for _ in ConeTypes]
                matched_cones_input[ConeTypes.LEFT] = sorted_left
                matched_cones_input[ConeTypes.RIGHT] = sorted_right

                cone_matching_input = ConeMatchingInput(
                    matched_cones_input, vehicle_position, vehicle_direction
                )
                self.cone_matching.set_new_input(cone_matching_input)
                (
                    left_cones_with_virtual,
                    right_cones_with_virtual,
                    left_to_right_match,
                    right_to_left_match,
                ) = self.cone_matching.run_cone_matching()

        with Timer("Path calculation", noprint=noprint):
            path_calculation_input = PathCalculationInput(
                left_cones_with_virtual,
                right_cones_with_virtual,
                left_to_right_match,
                right_to_left_match,
                vehicle_position,
                vehicle_direction,
                self.global_path,
            )
            self.pathing.set_new_input(path_calculation_input)
            final_path, _ = self.pathing.run_path_calculation()

            # 在 path_calculation 結束後打印結果
            rospy.loginfo(f"Calculated path start point: {final_path[0]}")

            '''
            # 如果起點偏移，強制重設為 (0, 0)
            if not np.allclose(final_path[0][:2], [0.0, 0.0], atol=1e-3):
                rospy.logwarn("Path start point deviates from (0, 0). Resetting to origin.")
                final_path[0][:2] = [0.0, 0.0]
            '''
            

        if self.relocalization_info is not None and self.relocalizer.is_relocalized:
            final_path = final_path.copy()
            path_xy = final_path[:, 1:3]
            fake_yaw = np.zeros(len(path_xy))

            path_xy, _ = self.relocalizer.transform_to_original_frame(path_xy, fake_yaw)

            final_path = final_path.copy()

            final_path[:, 1:3] = path_xy

        if return_intermediate_results:
            return (
                final_path,
                sorted_left,
                sorted_right,
                left_cones_with_virtual,
                right_cones_with_virtual,
                left_to_right_match,
                right_to_left_match,
            )

        return final_path

    @property
    def relocalization_info(self) -> RelocalizationInformation | None:
        if self.relocalizer is None:
            return None

        if not self.relocalizer.is_relocalized:
            return None

        return RelocalizationInformation.from_transform_function(
            self.relocalizer.transform_to_known_map_frame
        )


#########
# 全局變數
global cone_sorting_input
global vehicle_position
global vehicle_direction  # 新增方向變數

def plot_path(final_path, cones): #用Matplotlib顯示（不是rviz)
    plt.ion()  # 開啟互動模式
    plt.clf()  # 清除之前的圖形

    path_x = final_path[:, 1]
    path_y = final_path[:, 2]

    #plt.figure(figsize=(10, 10))  #創建新的圖框

    cones_x = cones[:, 0]
    cones_y = cones[:, 1]
    plt.scatter(cones_x, cones_y, color='black', label='Cones', zorder=5)
    
    # 繪製計算出的路徑
    plt.plot(path_x, path_y, '-o', color='blue', label='Path', zorder=10)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Path Planning with Cones')
    plt.legend()
    plt.grid(True)

    #plt.show(block=False)
    #plt.pause(1)  # 暫停一點時間以刷新顯示
    plt.draw()  # 更新圖形
    plt.pause(0.001)  # 短暫暫停，允許圖形更新


#def update_vehicle_position_and_direction(movement_delta):
    #global vehicle_position, vehicle_direction
    #previous_position = vehicle_position.copy()  # 儲存先前位置
    #vehicle_position = vehicle_position.astype(np.float64)  # 確保是 float64 類型
    #vehicle_position += movement_delta.astype(np.float64)  # 更新位置為 float64

''' #用Matplotlib顯示（不是rviz)

    # 計算方向向量
    vehicle_direction = vehicle_position - previous_position

    # 檢查是否為零向量，如果是，設置為默認方向
    if np.linalg.norm(vehicle_direction) == 0:
        vehicle_direction = np.array([0.0, 1.0])  # 預設為 X 軸方向
    else:
        vehicle_direction /= np.linalg.norm(vehicle_direction)  # 正規化方向
'''

def create_cone_marker(cone_position, cone_id, color):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.id = cone_id
    marker.pose.position.x = cone_position[0]
    marker.pose.position.y = cone_position[1]
    marker.pose.position.z = 0.2
    marker.scale.x = 0.5
    marker.scale.y = 0.5
    marker.scale.z = 0.5
    marker.color.a = 1.0  # 不透明度
    marker.color.r, marker.color.g, marker.color.b = color
    return marker

def create_path_marker(path_points):
    # 打印起點座標
    if len(path_points) > 0:
        start_point = path_points[0]
        rospy.loginfo(f"Path start point: x={start_point[0]}, y={start_point[1]}")

    marker = Marker()
    marker.header.frame_id = "map"
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.id = 1000  # 假設ID 1000留給路徑
    marker.scale.x = 0.1  # 線的寬度
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0
    marker.color.a = 1.0  # 不透明度
    
    for point in path_points:
        p = geometry_msgs.msg.Point()
        p.x = point[0]

        p.y = point[1]
        p.z = 0.0
        marker.points.append(p)
    
    return marker

'''
# 回調函數處理 YOLO 偵測結果
def yolo_callback(data): #用Matplotlib顯示（不是rviz)
    try:
        global cone_sorting_input
        global planner
        global vehicle_position
        global vehicle_direction

        # 模擬車輛移動
        #movement_delta = np.array([0.0, -0.1], dtype=np.float64)
        #update_vehicle_position_and_direction(movement_delta)  # 更新位置與方向

        # 確保方向不為零向量
        if np.linalg.norm(vehicle_direction) == 0:
            vehicle_direction = np.array([0, 1], dtype=np.float64)  # 設置為默認方向
        
        cones_world = np.array(list(zip(data.x, data.y)))
        cones_2d_relative = cones_world - vehicle_position

        # 圓錐分類
        cones_by_type = {ConeTypes.UNKNOWN: [], ConeTypes.RIGHT: [], ConeTypes.LEFT: []}
        for i, label in enumerate(data.labels):
            cone_type = classify_cone(label)
            cones_by_type[cone_type].append(cones_2d_relative[i])

        # 將圓錐分類轉換為 numpy 數組
        cones_by_type_array = [
            np.array(cones_by_type[ConeTypes.UNKNOWN]) if cones_by_type[ConeTypes.UNKNOWN] else np.zeros((0, 2)),
            np.array(cones_by_type[ConeTypes.RIGHT]) if cones_by_type[ConeTypes.RIGHT] else np.zeros((0, 2)),
            np.array(cones_by_type[ConeTypes.LEFT]) if cones_by_type[ConeTypes.LEFT] else np.zeros((0, 2))
        ]
        
        # 更新 cone_sorting_input 的數據
        cone_sorting_input.slam_position = vehicle_position
        cone_sorting_input.slam_direction = vehicle_direction
        cone_sorting_input.slam_cones = cones_by_type_array

        # 確認圓錐數據有效後，進行路徑計算
        final_path = planner.calculate_path_in_global_frame(
            cones=cones_2d_relative,
            vehicle_position=cone_sorting_input.slam_position,
            vehicle_direction=cone_sorting_input.slam_direction,
            return_intermediate_results=False
        )

        # 可視化並打印圓錐的排序結果
        plot_path(final_path, cones_2d_relative)

    except Exception as e:
        rospy.logerr(f"Error in yolo_callback: {e}")
'''        

# 建立 path 和 marker 發佈器
path_pub = rospy.Publisher("/calculated_path", Path, queue_size=10)
marker_pub = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size=10)

# 全局變數
callback_count = 0
start_time = None
latest_path = None
vehicle_position = np.array([0, 0], dtype=np.float64)  # 初始車輛位置
vehicle_direction = np.array([0, 1], dtype=np.float64)  # 初始車輛方向
previous_cone_marker_ids = []
data_queue = queue.Queue()  # 用於共享資料的隊列

# 刪除舊的圓錐 Marker
def delete_previous_cone_markers():
    global previous_cone_marker_ids
    marker_array = MarkerArray()
    for marker_id in previous_cone_marker_ids:
        delete_marker = Marker()
        delete_marker.header.frame_id = "map"
        delete_marker.id = marker_id
        delete_marker.action = Marker.DELETE
        marker_array.markers.append(delete_marker)
    marker_pub.publish(marker_array)
    rospy.sleep(0.005)
    previous_cone_marker_ids = []

# 發佈路徑

def publish_path(final_path):
    rospy.loginfo("Publishing path...")
    path_msg = Path()
    path_msg.header.stamp = rospy.Time.now()
    path_msg.header.frame_id = "map"
    for point in final_path:
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"
        pose.pose.position.x = point[1]
        pose.pose.position.y = point[2]
        pose.pose.position.z = 0.0
        path_msg.poses.append(pose)
    path_pub.publish(path_msg)

# 獨立執行緒處理路徑規劃邏輯
def calculate_path_worker():
    global latest_path
    
    # 確保方向不為零向量
    if np.linalg.norm(vehicle_direction) == 0:
        vehicle_direction = np.array([0, 1], dtype=np.float64)  # 設置為默認方向

    while not rospy.is_shutdown():
        try:
            cones_2d_relative, position, direction = data_queue.get(timeout=1)
            latest_path = planner.calculate_path_in_global_frame(
                cones=cones_2d_relative,
                vehicle_position=position,
                vehicle_direction=direction,
                return_intermediate_results=False
            )
            rospy.loginfo("Path calculated successfully.")
            publish_path(latest_path)
        except queue.Empty:
            continue
        except Exception as e:
            rospy.logerr(f"Error in calculate_path_worker: {e}")

# 背景執行緒啟動
path_thread = threading.Thread(target=calculate_path_worker, daemon=True)
path_thread.start()

# 顯示圓錐 Marker
def publish_cone_markers(cones, labels):
    global previous_cone_marker_ids
    marker_array = MarkerArray()
    for i, (cone, label) in enumerate(zip(cones, labels)):
        cone_type, color = classify_cone(label)
        marker = create_cone_marker(cone, i, color)
        marker_array.markers.append(marker)
        previous_cone_marker_ids.append(i)
    marker_pub.publish(marker_array)

# 回調函數

def yolo_callback(data):
    global callback_count, start_time, vehicle_position, vehicle_direction
    try:
        # 初始化計時器
        if start_time is None:
            start_time = time.time()

        # 處理 YOLO 偵測資料
        cones_world = np.array(list(zip(data.x, data.y)))
        cones_2d_relative = cones_world - vehicle_position

        # 刪除舊 Marker
        delete_previous_cone_markers()

        # 發佈圓錐 Marker
        publish_cone_markers(cones_2d_relative, data.labels)

        # 將資料放入隊列
        data_queue.put((cones_2d_relative, vehicle_position, vehicle_direction))

        # 更新回調計數
        callback_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            hz = callback_count / elapsed_time
            rospy.loginfo(f"Current processing speed: {hz:.2f} Hz")
    except Exception as e:
        rospy.logerr(f"Error in yolo_callback: {e}")

# 類別標籤分類
def classify_cone(label: str):
    if label == "yellow_cone":
        return ConeTypes.YELLOW, (1.0, 1.0, 0.0)
    elif label == "blue_cone":
        return ConeTypes.BLUE, (0.0, 0.0, 1.0)
    elif label == "orange_cone":
        return ConeTypes.ORANGE, (1.0, 0.0, 0.0)
    else:
        return ConeTypes.UNKNOWN, (0.5, 0.5, 0.5)

# 創建圓錐 Marker
def create_cone_marker(cone_position, cone_id, color):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.id = cone_id
    marker.pose.position.x = cone_position[0]
    marker.pose.position.y = cone_position[1]
    marker.pose.position.z = 0.2
    marker.scale.x = 0.5
    marker.scale.y = 0.5
    marker.scale.z = 0.5
    marker.color.a = 1.0
    marker.color.r, marker.color.g, marker.color.b = color
    return marker

# 主程式
if __name__ == '__main__':
    rospy.init_node('path_planning_with_yolo')
    planner = PathPlanner(MissionTypes.autocross)

    rospy.Subscriber("/yolo/objects/relative_coordinates", LabeledPointArray, yolo_callback)
    rospy.spin()