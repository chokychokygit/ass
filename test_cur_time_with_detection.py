# -*-coding:utf-8-*-
# INTEGRATED VERSION: test_cur_time_inter.py + fire_target.py
# รวมระบบสำรวจแผนที่ + ตรวจจับและยิงวัตถุอัตโนมัติ

import time
import robomaster
from robomaster import robot, camera as r_camera, blaster as r_blaster
import numpy as np
import math
import json
import matplotlib.pyplot as plt
from collections import deque
import traceback
import statistics
import cv2
import threading
import queue

# =============================================================================
# ===== CONFIGURATION & PARAMETERS ============================================
# =============================================================================
SPEED_ROTATE = 500
SAVE_PATH = r"D:\downsyndrome\year2_1\Robot_Module_2-1\Assignment\dude"

# --- Sharp Distance Sensor Configuration ---
LEFT_SHARP_SENSOR_ID = 1
LEFT_SHARP_SENSOR_PORT = 1
LEFT_TARGET_CM = 13.0

RIGHT_SHARP_SENSOR_ID = 2
RIGHT_SHARP_SENSOR_PORT = 1
RIGHT_TARGET_CM = 13.0

# --- IR Sensor Configuration ---
LEFT_IR_SENSOR_ID = 1
LEFT_IR_SENSOR_PORT = 2
RIGHT_IR_SENSOR_ID = 2
RIGHT_IR_SENSOR_PORT = 2

# --- Sharp Sensor Detection Thresholds ---
SHARP_WALL_THRESHOLD_CM = 60.0
SHARP_STDEV_THRESHOLD = 0.3

# --- ToF Centering Configuration ---
TOF_ADJUST_SPEED = 0.1
TOF_CALIBRATION_SLOPE = 0.0894
TOF_CALIBRATION_Y_INTERCEPT = 3.8409

# --- Logical state for the grid map ---
CURRENT_POSITION = (3,0)
CURRENT_DIRECTION = 1
TARGET_DESTINATION = (3,0)

# --- Physical state for the robot ---
CURRENT_TARGET_YAW = 0.0
ROBOT_FACE = 1

# --- IMU Drift Compensation Parameters ---
IMU_COMPENSATION_START_NODE_COUNT = 7
IMU_COMPENSATION_NODE_INTERVAL = 10
IMU_COMPENSATION_DEG_PER_INTERVAL = -2.0
IMU_DRIFT_COMPENSATION_DEG = 0.0

# --- Occupancy Grid Parameters ---
PROB_OCC_GIVEN_OCC = {'tof': 0.95, 'sharp': 0.90}
PROB_OCC_GIVEN_FREE = {'tof': 0.05, 'sharp': 0.10}

LOG_ODDS_OCC = {
    'tof': math.log(PROB_OCC_GIVEN_OCC['tof'] / (1 - PROB_OCC_GIVEN_OCC['tof'])),
    'sharp': math.log(PROB_OCC_GIVEN_OCC['sharp'] / (1 - PROB_OCC_GIVEN_OCC['sharp']))
}
LOG_ODDS_FREE = {
    'tof': math.log(PROB_OCC_GIVEN_FREE['tof'] / (1 - PROB_OCC_GIVEN_FREE['tof'])),
    'sharp': math.log(PROB_OCC_GIVEN_FREE['sharp'] / (1 - PROB_OCC_GIVEN_FREE['sharp']))
}

# --- Decision Thresholds ---
OCCUPANCY_THRESHOLD = 0.7
FREE_THRESHOLD = 0.3

# --- Timestamp Logging ---
POSITION_LOG = []

# =============================================================================
# ===== OBJECT DETECTION CONFIGURATION ========================================
# =============================================================================
# Target Configuration
TARGET_SHAPE = "Circle"
TARGET_COLOR = "Red"

# PID Parameters
PID_KP = -0.25
PID_KI = -0.01
PID_KD = -0.03
DERIV_LPF_ALPHA = 0.25

MAX_YAW_SPEED  = 220
MAX_PITCH_SPEED= 180
I_CLAMP = 2000.0

PIX_ERR_DEADZONE = 6
LOCK_TOL_X = 8
LOCK_TOL_Y = 8
LOCK_STABLE_COUNT = 6

# Camera Configuration
FRAME_W, FRAME_H = 960, 540
VERTICAL_FOV_DEG = 54.0
PIXELS_PER_DEG_V = FRAME_H / VERTICAL_FOV_DEG

# Pitch Bias
PITCH_BIAS_DEG = 2.5
PITCH_BIAS_PIX = +PITCH_BIAS_DEG * PIXELS_PER_DEG_V

# ROI Configuration
ROI_Y0, ROI_H0, ROI_X0, ROI_W0 = 264, 270, 10, 911
ROI_SHIFT_PER_DEG = 6.0
ROI_Y_MIN, ROI_Y_MAX = 0, FRAME_H - 10

# GPU check
USE_GPU = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("✅ CUDA available, enabling GPU path")
        USE_GPU = True
    else:
        print("⚠️ CUDA not available, CPU path")
except Exception:
    print("⚠️ Skip CUDA check, CPU path")

# =============================================================================
# ===== SHARED VARIABLES & THREADING ==========================================
# =============================================================================
frame_queue = queue.Queue(maxsize=1)
processed_output = {"details": []}
output_lock = threading.Lock()
stop_event = threading.Event()

# Gimbal angles
gimbal_angle_lock = threading.Lock()
gimbal_angles = (0.0, 0.0, 0.0, 0.0)

# Detection control
is_detecting_flag = {"v": False}  # เริ่มต้นปิด

# Detected objects log
detected_objects_log = []
detected_objects_lock = threading.Lock()

def sub_angle_cb(angle_info):
    global gimbal_angles
    with gimbal_angle_lock:
        gimbal_angles = tuple(angle_info)

# =============================================================================
# ===== HELPER FUNCTIONS ======================================================
# =============================================================================
def convert_adc_to_cm(adc_value):
    if adc_value <= 0: return float('inf')
    return 30263 * (adc_value ** -1.352)

def calibrate_tof_value(raw_tof_value):
    try:
        if raw_tof_value is None or raw_tof_value <= 0:
            return float('inf')
        return (TOF_CALIBRATION_SLOPE * raw_tof_value) + TOF_CALIBRATION_Y_INTERCEPT
    except Exception:
        return float('inf')

def get_compensated_target_yaw():
    return CURRENT_TARGET_YAW + IMU_DRIFT_COMPENSATION_DEG

def log_position_timestamp(position, direction, action="arrived"):
    global POSITION_LOG
    timestamp = time.time()
    direction_names = ['North', 'East', 'South', 'West']
    
    dt = time.gmtime(timestamp)
    microseconds = int((timestamp % 1) * 1000000)
    iso_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", dt) + f".{microseconds:06d}Z"
    
    log_entry = {
        "timestamp": timestamp,
        "iso_timestamp": iso_timestamp,
        "position": list(position),
        "direction": direction_names[direction],
        "action": action,
        "yaw_angle": CURRENT_TARGET_YAW,
        "imu_compensation": IMU_DRIFT_COMPENSATION_DEG
    }
    
    POSITION_LOG.append(log_entry)
    print(f"📍 [{action}] Position: {position}, Direction: {direction_names[direction]}, Time: {log_entry['iso_timestamp']}")

def log_detected_object(obj_info, robot_position):
    """บันทึกวัตถุที่ตรวจพบ"""
    global detected_objects_log
    
    timestamp = time.time()
    dt = time.gmtime(timestamp)
    microseconds = int((timestamp % 1) * 1000000)
    iso_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", dt) + f".{microseconds:06d}Z"
    
    log_entry = {
        "timestamp": timestamp,
        "iso_timestamp": iso_timestamp,
        "position": list(robot_position),
        "color": obj_info.get("color", "Unknown"),
        "shape": obj_info.get("shape", "Unknown"),
        "is_target": obj_info.get("is_target", False),
        "fired": False  # จะอัพเดทเป็น True เมื่อยิง
    }
    
    with detected_objects_lock:
        detected_objects_log.append(log_entry)
    
    print(f"🎯 Detected: {log_entry['color']} {log_entry['shape']} at {robot_position}")

# =============================================================================
# ===== OBJECT DETECTION FUNCTIONS ============================================
# =============================================================================
def apply_awb(bgr):
    if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "createLearningBasedWB"):
        wb = cv2.xphoto.createLearningBasedWB()
        try:
            wb.setSaturationThreshold(0.99)
        except Exception:
            pass
        return wb.balanceWhite(bgr)
    return bgr

def night_enhance_pipeline_cpu(bgr):
    return apply_awb(bgr)

class ObjectTracker:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        print(f"🖼️  ObjectTracker in {'GPU' if use_gpu else 'CPU'} mode")

    def _get_angle(self, pt1, pt2, pt0):
        dx1 = pt1[0] - pt0[0]; dy1 = pt1[1] - pt0[1]
        dx2 = pt2[0] - pt0[0]; dy2 = pt2[1] - pt0[1]
        dot = dx1*dx2 + dy1*dy2
        mag1 = (dx1*dx1 + dy1*dy1)**0.5
        mag2 = (dx2*dx2 + dy2*dy2)**0.5
        if mag1*mag2 == 0:
            return 0
        return math.degrees(math.acos(max(-1, min(1, dot/(mag1*mag2)))) )

    def get_raw_detections(self, frame):
        enhanced = cv2.GaussianBlur(night_enhance_pipeline_cpu(frame), (5,5), 0)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

        ranges = {
            'Red': ([0,80,40],[10,255,255],[170,80,40],[180,255,255]),
            'Yellow': ([20,60,40],[35,255,255]),
            'Green': ([35,40,30],[85,255,255]),
            'Blue': ([90,40,30],[130,255,255])
        }
        masks = {}
        masks['Red'] = cv2.inRange(hsv, np.array(ranges['Red'][0]), np.array(ranges['Red'][1])) | \
                       cv2.inRange(hsv, np.array(ranges['Red'][2]), np.array(ranges['Red'][3]))
        for name in ['Yellow','Green','Blue']:
            masks[name] = cv2.inRange(hsv, np.array(ranges[name][0]), np.array(ranges[name][1]))

        combined = masks['Red'] | masks['Yellow'] | masks['Green'] | masks['Blue']
        kernel = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)

        contours,_ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        H,W = frame.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500: continue
            x,y,w,h = cv2.boundingRect(cnt)
            if w==0 or h==0: continue
            ar = w/float(h)
            if ar>4.0 or ar<0.25: continue
            hull = cv2.convexHull(cnt); ha = cv2.contourArea(hull)
            if ha==0: continue
            solidity = area/ha
            if solidity < 0.85: continue
            if x<=2 or y<=2 or x+w>=W-2 or y+h>=H-2: continue

            contour_mask = np.zeros((H,W), np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
            max_mean, found = 0, "Unknown"
            for cname, m in masks.items():
                mv = cv2.mean(m, mask=contour_mask)[0]
                if mv > max_mean:
                    max_mean, found = mv, cname
            if max_mean <= 20: continue

            shape = "Uncertain"
            peri = cv2.arcLength(cnt, True)
            circ = (4*math.pi*area)/(peri*peri) if peri>0 else 0
            if circ > 0.82:
                shape = "Circle"
            else:
                approx = cv2.approxPolyDP(cnt, 0.04*peri, True)
                if len(approx)==4 and solidity>0.88:
                    pts=[tuple(p[0]) for p in approx]
                    angs=[self._get_angle(pts[(i-1)%4], pts[(i+1)%4], p) for i,p in enumerate(pts)]
                    if all(70<=a<=110 for a in angs):
                        _,(rw,rh),_ = cv2.minAreaRect(cnt)
                        if min(rw,rh)>0:
                            ar2 = max(rw,rh)/min(rw,rh)
                            if 0.88<=ar2<=1.12: shape="Square"
                            elif w>h: shape="Rectangle_H"
                            else: shape="Rectangle_V"
            out.append({"contour":cnt,"shape":shape,"color":found,"box":(x,y,w,h)})
        return out

# =============================================================================
# ===== CONNECTION MANAGEMENT =================================================
# =============================================================================
class RMConnection:
    def __init__(self):
        self._lock = threading.Lock()
        self._robot = None
        self.connected = threading.Event()

    def connect(self):
        with self._lock:
            self._safe_close()
            print("🤖 Connecting to RoboMaster...")
            rb = robot.Robot()
            rb.initialize(conn_type="ap")
            rb.camera.start_video_stream(display=False, resolution=r_camera.STREAM_540P)
            try:
                rb.gimbal.sub_angle(freq=50, callback=sub_angle_cb)
            except Exception as e:
                print("Gimbal sub_angle error:", e)
            self._robot = rb
            self.connected.set()
            print("✅ RoboMaster connected & camera streaming")

            try:
                rb.gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()
            except Exception as e:
                print("Recenter error:", e)

    def _safe_close(self):
        if self._robot is not None:
            try:
                try: self._robot.camera.stop_video_stream()
                except Exception: pass
                try:
                    try: self._robot.gimbal.unsub_angle()
                    except Exception: pass
                except Exception: pass
                try: self._robot.close()
                except Exception: pass
            finally:
                self._robot = None
                self.connected.clear()
                print("🔌 Connection closed")

    def drop_and_reconnect(self):
        with self._lock:
            self._safe_close()

    def get_camera(self):
        with self._lock:
            return None if self._robot is None else self._robot.camera

    def get_gimbal(self):
        with self._lock:
            return None if self._robot is None else self._robot.gimbal

    def get_blaster(self):
        with self._lock:
            return None if self._robot is None else self._robot.blaster
    
    def get_robot(self):
        with self._lock:
            return self._robot

    def close(self):
        with self._lock:
            self._safe_close()

def reconnector_thread(manager: RMConnection):
    backoff = 1.0
    while not stop_event.is_set():
        if not manager.connected.is_set():
            try:
                manager.connect()
                backoff = 1.0
            except Exception as e:
                print(f"♻️ Reconnect failed: {e} (retry in {backoff:.1f}s)")
                time.sleep(backoff)
                backoff = min(backoff*1.6, 8.0)
                continue
        time.sleep(0.2)

# =============================================================================
# ===== THREADING FUNCTIONS ===================================================
# =============================================================================
def capture_thread_func(manager: RMConnection, q: queue.Queue):
    print("🚀 Capture thread started")
    fail = 0
    while not stop_event.is_set():
        if not manager.connected.is_set():
            time.sleep(0.1); continue
        cam = manager.get_camera()
        if cam is None:
            time.sleep(0.1); continue
        try:
            frame = cam.read_cv2_image(timeout=1.0)
            if frame is not None:
                if q.full():
                    try: q.get_nowait()
                    except queue.Empty: pass
                q.put(frame)
                fail = 0
            else:
                fail += 1
        except Exception as e:
            print(f"CRITICAL: camera read error: {e}")
            fail += 1

        if fail >= 10:
            print("⚠️ Too many camera errors → drop & reconnect")
            manager.drop_and_reconnect()
            try:
                while True: q.get_nowait()
            except queue.Empty:
                pass
            fail = 0
        time.sleep(0.005)
    print("🛑 Capture thread stopped")

def processing_thread_func(tracker: ObjectTracker, q: queue.Queue,
                           target_shape, target_color,
                           roi_state,
                           is_detecting_func):
    global processed_output
    print("🧠 Processing thread started.")

    while not stop_event.is_set():
        if not is_detecting_func():
            time.sleep(0.05); continue
        try:
            frame_to_process = q.get(timeout=1.0)

            with gimbal_angle_lock:
                pitch_deg = gimbal_angles[0]
            roi_y_dynamic = int(ROI_Y0 - (max(0.0, -pitch_deg) * ROI_SHIFT_PER_DEG))
            roi_y_dynamic = max(ROI_Y_MIN, min(ROI_Y_MAX, roi_y_dynamic))

            ROI_X, ROI_W = roi_state["x"], roi_state["w"]
            ROI_H = roi_state["h"]
            roi_state["y"] = roi_y_dynamic

            roi_frame = frame_to_process[roi_y_dynamic:roi_y_dynamic+ROI_H, ROI_X:ROI_X+ROI_W]
            detections = tracker.get_raw_detections(roi_frame)

            detailed_results = []
            divider1 = int(ROI_W*0.33)
            divider2 = int(ROI_W*0.66)

            object_id_counter = 1
            for d in detections:
                shape, color, (x,y,w,h) = d['shape'], d['color'], d['box']
                endx = x+w
                zone = "Center"
                if endx < divider1: zone = "Left"
                elif x >= divider2: zone = "Right"
                is_target = (shape == target_shape and color == target_color)

                detailed_results.append({
                    "id": object_id_counter,
                    "color": color,
                    "shape": shape,
                    "zone": zone,
                    "is_target": is_target,
                    "box": (x,y,w,h)
                })
                
                # Log detected objects
                if is_target:
                    log_detected_object(detailed_results[-1], CURRENT_POSITION)
                
                object_id_counter += 1

            with output_lock:
                processed_output = {"details": detailed_results}

        except queue.Empty:
            continue
        except Exception as e:
            print(f"CRITICAL: Processing error: {e}")
            time.sleep(0.02)

    print("🛑 Processing thread stopped.")

def control_thread_func(manager: RMConnection, roi_state, is_detecting_func):
    print("🎯 Control thread started.")
    prev_time = None
    err_x_prev_f = 0.0
    err_y_prev_f = 0.0
    integ_x = 0.0
    integ_y = 0.0

    lock_queue = deque(maxlen=LOCK_STABLE_COUNT)

    while not stop_event.is_set():
        if not (is_detecting_func() and manager.connected.is_set()):
            time.sleep(0.02); continue

        with output_lock:
            dets = list(processed_output["details"])

        target_box = None
        max_area = -1
        target_det = None
        for det in dets:
            if det.get("is_target", False):
                x,y,w,h = det["box"]
                area = w*h
                if area > max_area:
                    max_area = area
                    target_box = (x,y,w,h)
                    target_det = det

        gimbal = manager.get_gimbal()
        blaster = manager.get_blaster()
        if (gimbal is None) or (blaster is None):
            time.sleep(0.02); continue

        if target_box is not None:
            x,y,w,h = target_box

            cx_roi = x + w/2.0
            cy_roi = y + h/2.0

            ROI_X, ROI_Y, ROI_W, ROI_H = roi_state["x"], roi_state["y"], roi_state["w"], roi_state["h"]
            cx = ROI_X + cx_roi
            cy = ROI_Y + cy_roi

            center_x = FRAME_W/2.0
            center_y = FRAME_H/2.0

            err_x = (center_x - cx)
            err_y = (center_y - cy) + PITCH_BIAS_PIX

            if abs(err_x) < PIX_ERR_DEADZONE: err_x = 0.0
            if abs(err_y) < PIX_ERR_DEADZONE: err_y = 0.0

            now = time.time()
            if prev_time is None:
                prev_time = now
                err_x_prev_f = err_x
                err_y_prev_f = err_y
                time.sleep(0.005)
                continue
            dt = max(1e-3, now - prev_time)
            prev_time = now

            err_x_f = err_x_prev_f + DERIV_LPF_ALPHA*(err_x - err_x_prev_f)
            err_y_f = err_y_prev_f + DERIV_LPF_ALPHA*(err_y - err_y_prev_f)
            dx = (err_x_f - err_x_prev_f)/dt
            dy = (err_y_f - err_y_prev_f)/dt
            err_x_prev_f = err_x_f
            err_y_prev_f = err_y_f

            integ_x = np.clip(integ_x + err_x*dt, -I_CLAMP, I_CLAMP)
            integ_y = np.clip(integ_y + err_y*dt, -I_CLAMP, I_CLAMP)

            u_x = PID_KP*err_x + PID_KI*integ_x + PID_KD*dx
            u_y = PID_KP*err_y + PID_KI*integ_y + PID_KD*dy

            u_x = float(np.clip(u_x, -MAX_YAW_SPEED, MAX_YAW_SPEED))
            u_y = float(np.clip(u_y, -MAX_PITCH_SPEED, MAX_PITCH_SPEED))

            try:
                gimbal.drive_speed(pitch_speed=-u_y, yaw_speed=u_x)
            except Exception as e:
                print("drive_speed error:", e)

            locked = (abs(err_x) <= LOCK_TOL_X) and (abs(err_y) <= LOCK_TOL_Y)
            lock_queue.append(1 if locked else 0)

            if len(lock_queue) == LOCK_STABLE_COUNT and sum(lock_queue) == LOCK_STABLE_COUNT:
                try:
                    print(f"🔥 FIRING at {target_det['color']} {target_det['shape']}!")
                    blaster.fire(fire_type=r_blaster.WATER_FIRE)
                    
                    # Update log
                    with detected_objects_lock:
                        for obj in detected_objects_log:
                            if (obj["position"] == list(CURRENT_POSITION) and 
                                obj["color"] == target_det["color"] and 
                                obj["shape"] == target_det["shape"] and
                                not obj["fired"]):
                                obj["fired"] = True
                                break
                    
                    time.sleep(0.1)
                    lock_queue.clear()
                except Exception as e:
                    print("fire error:", e)
        else:
            try:
                gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            except Exception:
                pass
            lock_queue.clear()
            integ_x *= 0.98
            integ_y *= 0.98

        time.sleep(0.005)

    print("🛑 Control thread stopped.")

# =============================================================================
# ===== OCCUPANCY GRID MAP & VISUALIZATION ====================================
# =============================================================================
class WallBelief:
    def __init__(self):
        self.log_odds = 0.0

    def update(self, is_occupied_reading, sensor_type):
        if is_occupied_reading:
            self.log_odds += LOG_ODDS_OCC[sensor_type]
        else:
            self.log_odds += LOG_ODDS_FREE[sensor_type]
        self.log_odds = max(min(self.log_odds, 10), -10)

    def get_probability(self):
        return 1.0 - 1.0 / (1.0 + math.exp(self.log_odds))

    def is_occupied(self):
        return self.get_probability() > OCCUPANCY_THRESHOLD

class OGMCell:
    def __init__(self):
        self.log_odds_occupied = 0.0
        self.walls = {'N': None, 'E': None, 'S': None, 'W': None}

    def get_node_probability(self):
        return 1.0 - 1.0 / (1.0 + math.exp(self.log_odds_occupied))

    def is_node_occupied(self):
        return self.get_node_probability() > OCCUPANCY_THRESHOLD

class OccupancyGridMap:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[OGMCell() for _ in range(width)] for _ in range(height)]
        self._link_walls()

    def _link_walls(self):
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c].walls['N'] is None:
                    wall = WallBelief()
                    self.grid[r][c].walls['N'] = wall
                    if r > 0: self.grid[r-1][c].walls['S'] = wall
                if self.grid[r][c].walls['W'] is None:
                    wall = WallBelief()
                    self.grid[r][c].walls['W'] = wall
                    if c > 0: self.grid[r][c-1].walls['E'] = wall
                if self.grid[r][c].walls['S'] is None:
                    self.grid[r][c].walls['S'] = WallBelief()
                if self.grid[r][c].walls['E'] is None:
                    self.grid[r][c].walls['E'] = WallBelief()

    def update_wall(self, r, c, direction_char, is_occupied_reading, sensor_type):
        if 0 <= r < self.height and 0 <= c < self.width:
            wall = self.grid[r][c].walls.get(direction_char)
            if wall:
                wall.update(is_occupied_reading, sensor_type)

    def update_node(self, r, c, is_occupied_reading, sensor_type='tof'):
        if 0 <= r < self.height and 0 <= c < self.width:
            if is_occupied_reading:
                self.grid[r][c].log_odds_occupied += LOG_ODDS_OCC[sensor_type]
            else:
                self.grid[r][c].log_odds_occupied += LOG_ODDS_FREE[sensor_type]

    def is_path_clear(self, r1, c1, r2, c2):
        dr, dc = r2 - r1, c2 - c1
        if abs(dr) + abs(dc) != 1: return False
        if dr == -1: wall_char = 'N'
        elif dr == 1: wall_char = 'S'
        elif dc == 1: wall_char = 'E'
        elif dc == -1: wall_char = 'W'
        else: return False
        wall = self.grid[r1][c1].walls.get(wall_char)
        if wall and wall.is_occupied(): return False
        if 0 <= r2 < self.height and 0 <= c2 < self.width:
            if self.grid[r2][c2].is_node_occupied(): return False
        else: return False
        return True

class RealTimeVisualizer:
    def __init__(self, grid_size, target_dest=None):
        self.grid_size = grid_size
        self.target_dest = target_dest
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        self.colors = {"robot": "#0000FF", "target": "#FFD700", "path": "#FFFF00", "wall": "#000000", "wall_prob": "#000080"}

    def update_plot(self, occupancy_map, robot_pos, path=None):
        self.ax.clear()
        self.ax.set_title("Real-time Hybrid Belief Map (Nodes & Walls)")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(self.grid_size - 0.5, -0.5)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                prob = occupancy_map.grid[r][c].get_node_probability()
                if prob > OCCUPANCY_THRESHOLD: color = '#8B0000'
                elif prob < FREE_THRESHOLD: color = '#D3D3D3'
                else: color = '#90EE90'
                self.ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor='k', lw=0.5))
                self.ax.text(c, r, f"{prob:.2f}", ha="center", va="center", color="black", fontsize=8)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = occupancy_map.grid[r][c]
                prob_n = cell.walls['N'].get_probability()
                if abs(prob_n - 0.5) > 0.01: self.ax.text(c, r - 0.5, f"{prob_n:.2f}", ha="center", va="center", color=self.colors['wall_prob'], fontsize=6, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1', edgecolor='none'))
                prob_w = cell.walls['W'].get_probability()
                if abs(prob_w - 0.5) > 0.01: self.ax.text(c - 0.5, r, f"{prob_w:.2f}", ha="center", va="center", color=self.colors['wall_prob'], fontsize=6, rotation=90, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1', edgecolor='none'))
        for c in range(self.grid_size):
            r_edge = self.grid_size - 1
            prob_s = occupancy_map.grid[r_edge][c].walls['S'].get_probability()
            if abs(prob_s - 0.5) > 0.01: self.ax.text(c, r_edge + 0.5, f"{prob_s:.2f}", ha="center", va="center", color=self.colors['wall_prob'], fontsize=6, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1', edgecolor='none'))
        for r in range(self.grid_size):
            c_edge = self.grid_size - 1
            prob_e = occupancy_map.grid[r][c_edge].walls['E'].get_probability()
            if abs(prob_e - 0.5) > 0.01: self.ax.text(c_edge + 0.5, r, f"{prob_e:.2f}", ha="center", va="center", color=self.colors['wall_prob'], fontsize=6, rotation=90, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1', edgecolor='none'))
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = occupancy_map.grid[r][c]
                if cell.walls['N'].is_occupied(): self.ax.plot([c - 0.5, c + 0.5], [r - 0.5, r - 0.5], color=self.colors['wall'], linewidth=4)
                if cell.walls['W'].is_occupied(): self.ax.plot([c - 0.5, c - 0.5], [r - 0.5, r + 0.5], color=self.colors['wall'], linewidth=4)
                if r == self.grid_size - 1 and cell.walls['S'].is_occupied(): self.ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], color=self.colors['wall'], linewidth=4)
                if c == self.grid_size - 1 and cell.walls['E'].is_occupied(): self.ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], color=self.colors['wall'], linewidth=4)
        if self.target_dest:
            r_t, c_t = self.target_dest
            self.ax.add_patch(plt.Rectangle((c_t - 0.5, r_t - 0.5), 1, 1, facecolor=self.colors['target'], edgecolor='k', lw=2, alpha=0.8))
        if path:
            for r_p, c_p in path: self.ax.add_patch(plt.Rectangle((c_p - 0.5, r_p - 0.5), 1, 1, facecolor=self.colors['path'], edgecolor='k', lw=0.5, alpha=0.7))
        if robot_pos:
            r_r, c_r = robot_pos
            self.ax.add_patch(plt.Rectangle((c_r - 0.5, r_r - 0.5), 1, 1, facecolor=self.colors['robot'], edgecolor='k', lw=2))
        legend_elements = [ plt.Rectangle((0,0),1,1, facecolor='#8B0000', label=f'Node Occupied (P>{OCCUPANCY_THRESHOLD})'), plt.Rectangle((0,0),1,1, facecolor='#90EE90', label=f'Node Unknown'), plt.Rectangle((0,0),1,1, facecolor='#D3D3D3', label=f'Node Free (P<{FREE_THRESHOLD})'), plt.Line2D([0], [0], color=self.colors['wall'], lw=4, label='Wall Occupied'), plt.Rectangle((0,0),1,1, facecolor=self.colors['robot'], label='Robot'), plt.Rectangle((0,0),1,1, facecolor=self.colors['target'], label='Target') ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.55, 1.0))
        self.fig.tight_layout(rect=[0, 0, 0.75, 1])
        self.fig.canvas.draw(); self.fig.canvas.flush_events(); plt.pause(0.01)

# =============================================================================
# ===== CORE ROBOT CONTROL CLASSES ============================================
# =============================================================================
class AttitudeHandler:
    def __init__(self):
        self.current_yaw, self.yaw_tolerance, self.is_monitoring = 0.0, 3.0, False
    def attitude_handler(self, attitude_info):
        if self.is_monitoring: self.current_yaw = attitude_info[0]
    def start_monitoring(self, chassis):
        self.is_monitoring = True; chassis.sub_attitude(freq=20, callback=self.attitude_handler)
    def stop_monitoring(self, chassis):
        self.is_monitoring = False;
        try: chassis.unsub_attitude()
        except Exception: pass
    def normalize_angle(self, angle):
        while angle > 180: angle -= 360
        while angle <= -180: angle += 360
        return angle
    def correct_yaw_to_target(self, chassis, target_yaw=0.0):
        normalized_target = self.normalize_angle(target_yaw); time.sleep(0.1)
        robot_rotation = -self.normalize_angle(normalized_target - self.current_yaw)
        print(f"\n🔧 Correcting Yaw: {self.current_yaw:.1f}° -> {target_yaw:.1f}°. Rotating: {robot_rotation:.1f}°")
        if abs(robot_rotation) > self.yaw_tolerance:
            chassis.move(x=0, y=0, z=robot_rotation, z_speed=60).wait_for_completed(timeout=2)
            time.sleep(0.1)
        final_error = abs(self.normalize_angle(normalized_target - self.current_yaw))
        if final_error <= self.yaw_tolerance: print(f"✅ Yaw Correction Success: {self.current_yaw:.1f}°"); return True
        print(f"⚠️ First attempt incomplete. Current: {self.current_yaw:.1f}°. Fine-tuning...")
        remaining_rotation = -self.normalize_angle(normalized_target - self.current_yaw)
        if abs(remaining_rotation) > 0.5 and abs(remaining_rotation) < 20:
            chassis.move(x=0, y=0, z=remaining_rotation, z_speed=40).wait_for_completed(timeout=2)
            time.sleep(0.1)
        final_error = abs(self.normalize_angle(normalized_target - self.current_yaw))
        if final_error <= self.yaw_tolerance: print(f"✅ Yaw Fine-tuning Success: {self.current_yaw:.1f}°"); return True
        else: print(f"🔥🔥 Yaw Correction FAILED. Final Yaw: {self.current_yaw:.1f}°"); return False

class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp, self.Ki, self.Kd, self.setpoint = Kp, Ki, Kd, setpoint
        self.prev_error, self.integral, self.integral_max = 0, 0, 1.0
    def compute(self, current, dt):
        error = self.setpoint - current
        self.integral += error * dt; self.integral = max(min(self.integral, self.integral_max), -self.integral_max)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error; return output

class MovementController:
    def __init__(self, chassis):
        self.chassis = chassis
        self.current_x_pos, self.current_y_pos = 0.0, 0.0
        self.chassis.sub_position(freq=20, callback=self.position_handler)
    def position_handler(self, position_info):
        self.current_x_pos, self.current_y_pos = position_info[0], position_info[1]

    def _calculate_yaw_correction(self, attitude_handler, target_yaw):
        KP_YAW = 1.8; MAX_YAW_SPEED = 25
        yaw_error = attitude_handler.normalize_angle(target_yaw - attitude_handler.current_yaw)
        speed = KP_YAW * yaw_error
        return max(min(speed, MAX_YAW_SPEED), -MAX_YAW_SPEED)

    def move_forward_one_grid(self, axis, attitude_handler):
        attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())
        target_distance = 0.6
        pid = PID(Kp=1.0, Ki=0.25, Kd=8, setpoint=target_distance)
        start_time, last_time = time.time(), time.time()
        start_position = self.current_x_pos if axis == 'x' else self.current_y_pos
        print(f"🚀 Moving FORWARD 0.6m, monitoring GLOBAL AXIS '{axis}'")
        while time.time() - start_time < 3.5:
            now = time.time(); dt = now - last_time; last_time = now
            current_position = self.current_x_pos if axis == 'x' else self.current_y_pos
            relative_position = abs(current_position - start_position)
            if abs(relative_position - target_distance) < 0.03:
                print("\n✅ Move complete!"); break
            output = pid.compute(relative_position, dt)
            ramp_multiplier = min(1.0, 0.1 + ((now - start_time) / 1.0) * 0.9)
            speed = max(-1.0, min(1.0, output * ramp_multiplier))
            yaw_correction = self._calculate_yaw_correction(attitude_handler, get_compensated_target_yaw())
            self.chassis.drive_speed(x=speed, y=0, z=yaw_correction, timeout=1)
            print(f"Moving... Dist: {relative_position:.3f}/{target_distance:.2f} m", end='\r')
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.5)

    def adjust_position_to_wall(self, sensor_adaptor, attitude_handler, side, sensor_config, target_distance_cm, direction_multiplier):
        compensated_yaw = get_compensated_target_yaw()
        print(f"\n--- Adjusting {side} Side (Yaw locked at {compensated_yaw:.2f}°) ---")
        print(f"   -> Config: ID={sensor_config['sharp_id']}, Port={sensor_config['sharp_port']}, Target={target_distance_cm}cm")
        TOLERANCE_CM, MAX_EXEC_TIME, KP_SLIDE, MAX_SLIDE_SPEED = 0.8, 8, 0.045, 0.18
        start_time = time.time()
        while time.time() - start_time < MAX_EXEC_TIME:
            adc_val = sensor_adaptor.get_adc(id=sensor_config["sharp_id"], port=sensor_config["sharp_port"])
            current_dist = convert_adc_to_cm(adc_val)
            dist_error = target_distance_cm - current_dist
            if abs(dist_error) <= TOLERANCE_CM:
                print(f"\n[{side}] Target distance reached! Final distance: {current_dist:.2f} cm")
                break
            slide_speed = max(min(direction_multiplier * KP_SLIDE * dist_error, MAX_SLIDE_SPEED), -MAX_SLIDE_SPEED)
            yaw_correction = self._calculate_yaw_correction(attitude_handler, compensated_yaw)
            self.chassis.drive_speed(x=0, y=slide_speed, z=yaw_correction)
            print(f"Adjusting {side}... Current: {current_dist:5.2f}cm, Target: {target_distance_cm:4.1f}cm, Error: {dist_error:5.2f}cm, Speed: {slide_speed:5.3f}", end='\r')
            time.sleep(0.02)
        else:
            print(f"\n[{side}] Movement timed out!")
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        time.sleep(0.1)

    def center_in_node_with_tof(self, scanner, attitude_handler, target_cm=19, tol_cm=1.0, max_adjust_time=6.0):
        if scanner.is_performing_full_scan:
            print("[ToF Centering] SKIPPED: A critical side-scan is in progress.")
            return

        print("\n--- Stage: Centering in Node with ToF ---")
        time.sleep(0.1)
        tof_dist = scanner.last_tof_distance_cm
        if tof_dist is None or math.isinf(tof_dist):
            print("[ToF] ❌ No valid ToF data available. Skipping centering.")
            return
        print(f"[ToF] Initial front distance: {tof_dist:.2f} cm")
        if tof_dist >= 50:
            print("[ToF] Distance >= 50cm, likely in an open space. Skipping centering.")
            return
        direction = 0
        if tof_dist > target_cm + tol_cm:
            print("[ToF] Too far from front wall. Moving forward...")
            direction = abs(TOF_ADJUST_SPEED)
        elif tof_dist < 22:
            print("[ToF] Too close to front wall. Moving backward...")
            direction = -abs(TOF_ADJUST_SPEED)
        else:
            print("[ToF] In range (22cm - target), but not centered. Moving forward...")
            direction = abs(TOF_ADJUST_SPEED)
        if direction == 0:
            print(f"[ToF] Already centered within tolerance ({tof_dist:.2f} cm). Skipping.")
            return
        start_time = time.time()
        compensated_yaw = get_compensated_target_yaw()
        while time.time() - start_time < max_adjust_time:
            yaw_correction = self._calculate_yaw_correction(attitude_handler, compensated_yaw)
            self.chassis.drive_speed(x=direction, y=0, z=yaw_correction, timeout=0.1)
            time.sleep(0.12)
            self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
            time.sleep(0.08)
            current_tof = scanner.last_tof_distance_cm
            if current_tof is None or math.isinf(current_tof):
                continue
            print(f"[ToF] Adjusting... Current Distance: {current_tof:.2f} cm", end="\r")
            if abs(current_tof - target_cm) <= tol_cm:
                print(f"\n[ToF] ✅ Centering complete. Final distance: {current_tof:.2f} cm")
                break
            if (direction > 0 and current_tof < target_cm - tol_cm) or \
            (direction < 0 and current_tof > target_cm + tol_cm):
                direction *= -1
                print("\n[ToF] 🔄 Overshot target. Reversing direction for fine-tuning.")
        else:
            print(f"\n[ToF] ⚠️ Centering timed out. Final distance: {scanner.last_tof_distance_cm:.2f} cm")
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        time.sleep(0.1)

    def rotate_to_direction(self, target_direction, attitude_handler):
        global CURRENT_DIRECTION
        if CURRENT_DIRECTION == target_direction: return
        diff = (target_direction - CURRENT_DIRECTION + 4) % 4
        if diff == 1: self.rotate_90_degrees_right(attitude_handler)
        elif diff == 3: self.rotate_90_degrees_left(attitude_handler)
        elif diff == 2: self.rotate_90_degrees_right(attitude_handler); self.rotate_90_degrees_right(attitude_handler)

    def rotate_90_degrees_right(self, attitude_handler):
        global CURRENT_TARGET_YAW, CURRENT_DIRECTION, ROBOT_FACE
        print("🔄 Rotating 90° RIGHT...")
        CURRENT_TARGET_YAW = attitude_handler.normalize_angle(CURRENT_TARGET_YAW + 90)
        attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())
        CURRENT_DIRECTION = (CURRENT_DIRECTION + 1) % 4; ROBOT_FACE += 1
    def rotate_90_degrees_left(self, attitude_handler):
        global CURRENT_TARGET_YAW, CURRENT_DIRECTION, ROBOT_FACE
        print("🔄 Rotating 90° LEFT...")
        CURRENT_TARGET_YAW = attitude_handler.normalize_angle(CURRENT_TARGET_YAW - 90)
        attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())
        CURRENT_DIRECTION = (CURRENT_DIRECTION - 1 + 4) % 4; ROBOT_FACE -= 1
        if ROBOT_FACE < 1: ROBOT_FACE += 4
    def cleanup(self):
        try: self.chassis.unsub_position()
        except Exception: pass

class EnvironmentScanner:
    def __init__(self, sensor_adaptor, tof_sensor, gimbal, chassis):
        self.sensor_adaptor, self.tof_sensor, self.gimbal, self.chassis = sensor_adaptor, tof_sensor, gimbal, chassis
        self.tof_wall_threshold_cm = 60.0
        
        self.last_tof_distance_cm = float('inf')
        self.side_tof_reading_cm = float('inf')
        self.is_gimbal_centered = True
        
        self.is_performing_full_scan = False

        self.tof_sensor.sub_distance(freq=10, callback=self._tof_data_handler)
        
        self.side_sensors = {
            "Left":  { "sharp_id": 1, "sharp_port": 1, "ir_id": 1, "ir_port": 2 },
            "Right": { "sharp_id": 2, "sharp_port": 1, "ir_id": 2, "ir_port": 2 }
        }

    def _tof_data_handler(self, sub_info):
        calibrated_cm = calibrate_tof_value(sub_info[0])
        if self.is_gimbal_centered:
            self.last_tof_distance_cm = calibrated_cm
        else:
            self.side_tof_reading_cm = calibrated_cm

    def _get_stable_reading_cm(self, side, duration=0.35):
        sensor_info = self.side_sensors.get(side)
        if not sensor_info: return None, None
        readings = []
        start_time = time.time()
        while time.time() - start_time < duration:
            adc = self.sensor_adaptor.get_adc(id=sensor_info["sharp_id"], port=sensor_info["sharp_port"])
            readings.append(convert_adc_to_cm(adc))
            time.sleep(0.05)
        if len(readings) < 5: return None, None
        return statistics.mean(readings), statistics.stdev(readings)

    def get_sensor_readings(self):
        self.is_performing_full_scan = True
        try:
            self.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed(); time.sleep(0.15)
            
            readings = {}
            readings['front'] = (self.last_tof_distance_cm < self.tof_wall_threshold_cm)
            print(f"[SCAN] Front (ToF): {self.last_tof_distance_cm:.1f}cm -> {'OCCUPIED' if readings['front'] else 'FREE'}")
            
            for side in ["Left", "Right"]:
                avg_dist, std_dev = self._get_stable_reading_cm(side)
                if avg_dist is None:
                    print(f"[{side.upper()}] Wall Check Error: Not enough sensor data.")
                    readings[side.lower()] = False
                    continue
                
                is_sharp_detecting_wall = (avg_dist < SHARP_WALL_THRESHOLD_CM and std_dev < SHARP_STDEV_THRESHOLD)
                ir_value = self.sensor_adaptor.get_io(id=self.side_sensors[side]["ir_id"], port=self.side_sensors[side]["ir_port"])
                is_ir_detecting_wall = (ir_value == 0)

                print(f"\n[SCAN] {side} Side Analysis:")
                print(f"    -> Sharp -> Suggests: {'WALL' if is_sharp_detecting_wall else 'FREE'}")
                print(f"    -> IR    -> Suggests: {'WALL' if is_ir_detecting_wall else 'FREE'}")

                if is_sharp_detecting_wall == is_ir_detecting_wall:
                    is_wall = is_sharp_detecting_wall
                    print(f"    -> Decision: Sensors agree. Result is {'WALL' if is_wall else 'FREE'}.")
                else:
                    print("    -> Ambiguity detected! Confirming with ToF...")
                    target_gimbal_yaw = -90 if side == "Left" else 90
                    
                    try:
                        self.is_gimbal_centered = False
                        self.gimbal.moveto(pitch=0, yaw=target_gimbal_yaw, yaw_speed=SPEED_ROTATE).wait_for_completed()
                        time.sleep(0.1)
                        
                        tof_confirm_dist_cm = self.side_tof_reading_cm
                        print(f"    -> ToF reading at {target_gimbal_yaw}° is {tof_confirm_dist_cm:.2f} cm.")
                        
                        is_wall = (tof_confirm_dist_cm < self.tof_wall_threshold_cm)
                        print(f"    -> ToF Confirmation: {'WALL DETECTED' if is_wall else 'NO WALL'}.")
                    
                    finally:
                        self.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
                        self.is_gimbal_centered = True
                        time.sleep(0.1)

                readings[side.lower()] = is_wall
                print(f"    -> Final Result for {side} side: {'WALL' if is_wall else 'FREE'}")
            
            return readings
        finally:
            self.is_performing_full_scan = False

    def get_front_tof_cm(self):
        self.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
        time.sleep(0.1)
        return self.last_tof_distance_cm

    def cleanup(self):
        try: self.tof_sensor.unsub_distance()
        except Exception: pass

# =============================================================================
# ===== PATHFINDING & EXPLORATION LOGIC =======================================
# =============================================================================
def find_path_bfs(occupancy_map, start, end):
    queue = deque([[start]]); visited = {start}
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    while queue:
        path = queue.popleft()
        r, c = path[-1]
        if (r, c) == end: return path
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < occupancy_map.height and 0 <= nc < occupancy_map.width:
                if occupancy_map.is_path_clear(r, c, nr, nc) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    new_path = list(path)
                    new_path.append((nr, nc))
                    queue.append(new_path)
    return None

def find_nearest_unvisited_path(occupancy_map, start_pos, visited_cells):
    h, w = occupancy_map.height, occupancy_map.width
    unvisited_cells_coords = []
    for r in range(h):
        for c in range(w):
            if (r, c) not in visited_cells and not occupancy_map.grid[r][c].is_node_occupied():
                unvisited_cells_coords.append((r, c))
    if not unvisited_cells_coords: return None
    shortest_path = None
    for target_pos in unvisited_cells_coords:
        path = find_path_bfs(occupancy_map, start_pos, target_pos)
        if path:
            if shortest_path is None or len(path) < len(shortest_path):
                shortest_path = path
    return shortest_path

def execute_path(path, movement_controller, attitude_handler, scanner, visualizer, occupancy_map, path_name="Backtrack"):
    global CURRENT_POSITION
    print(f"🎯 Executing {path_name} Path: {path}")
    dir_vectors_map = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    dir_map_abs_char = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
    
    log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, f"{path_name}_start")

    for i in range(len(path) - 1):
        visualizer.update_plot(occupancy_map, path[i], path)
        current_r, current_c = path[i]
        
        if i + 1 < len(path):
            next_r, next_c = path[i+1]
            dr, dc = next_r - current_r, next_c - current_c
            
            target_direction = dir_vectors_map[(dr, dc)]
            
            movement_controller.rotate_to_direction(target_direction, attitude_handler)
            
            print(f"   -> [{path_name}] Confirming path to ({next_r},{next_c}) with ToF...")
            scanner.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
            time.sleep(0.15)
            
            is_blocked = scanner.get_front_tof_cm() < scanner.tof_wall_threshold_cm
            
            occupancy_map.update_wall(current_r, current_c, dir_map_abs_char[CURRENT_DIRECTION], is_blocked, 'tof')
            print(f"   -> [{path_name}] Real-time ToF check: Path is {'BLOCKED' if is_blocked else 'CLEAR'}.")
            visualizer.update_plot(occupancy_map, CURRENT_POSITION)

            if is_blocked:
                print(f"   -> 🔥 [{path_name}] IMMEDIATE STOP. Real-time sensor detected an obstacle. Aborting path.")
                break

            axis_to_monitor = 'x' if ROBOT_FACE % 2 != 0 else 'y'
            movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)
            
            movement_controller.center_in_node_with_tof(scanner, attitude_handler)

            CURRENT_POSITION = (next_r, next_c)
            log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, f"{path_name}_moved")
            visualizer.update_plot(occupancy_map, CURRENT_POSITION, path)
            
            print(f"   -> [{path_name}] Performing side alignment at new position {CURRENT_POSITION}")
            perform_side_alignment_and_mapping(movement_controller, scanner, attitude_handler, occupancy_map, visualizer)

    print(f"✅ {path_name} complete.")

def perform_side_alignment_and_mapping(movement_controller, scanner, attitude_handler, occupancy_map, visualizer):
    print("\n--- Stage: Wall Detection & Side Alignment ---")
    r, c = CURRENT_POSITION
    dir_map_abs_char = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
    
    side_walls_present = scanner.get_sensor_readings()

    left_dir_abs = (CURRENT_DIRECTION - 1 + 4) % 4
    occupancy_map.update_wall(r, c, dir_map_abs_char[left_dir_abs], side_walls_present['left'], 'sharp')
    visualizer.update_plot(occupancy_map, CURRENT_POSITION)
    if side_walls_present['left']:
        movement_controller.adjust_position_to_wall(
            scanner.sensor_adaptor, attitude_handler, "Left", 
            scanner.side_sensors["Left"], LEFT_TARGET_CM, direction_multiplier=1
        )
    
    right_dir_abs = (CURRENT_DIRECTION + 1) % 4
    occupancy_map.update_wall(r, c, dir_map_abs_char[right_dir_abs], side_walls_present['right'], 'sharp')
    visualizer.update_plot(occupancy_map, CURRENT_POSITION)
    if side_walls_present['right']:
        movement_controller.adjust_position_to_wall(
            scanner.sensor_adaptor, attitude_handler, "Right", 
            scanner.side_sensors["Right"], RIGHT_TARGET_CM, direction_multiplier=-1
        )

    if not side_walls_present['left'] and not side_walls_present['right']:
        print("\n⚠️  WARNING: No side walls detected. Skipping alignment.")
    
    attitude_handler.correct_yaw_to_target(movement_controller.chassis, get_compensated_target_yaw())
    time.sleep(0.1)

def perform_object_detection_scan(scan_duration=3.0):
    """เปิดการตรวจจับวัตถุชั่วคราวเพื่อสแกนหาเป้าหมาย"""
    global is_detecting_flag
    
    print("\n🎯 === OBJECT DETECTION SCAN ACTIVATED ===")
    is_detecting_flag["v"] = True
    time.sleep(scan_duration)
    is_detecting_flag["v"] = False
    print("🛑 === OBJECT DETECTION SCAN DEACTIVATED ===\n")

def explore_with_ogm(scanner, movement_controller, attitude_handler, occupancy_map, visualizer, max_steps=40):
    global CURRENT_POSITION, CURRENT_DIRECTION, IMU_DRIFT_COMPENSATION_DEG
    visited_cells = set()
    
    log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, "exploration_start")
    
    for step in range(max_steps):
        r, c = CURRENT_POSITION
        print(f"\n--- Step {step + 1} at {CURRENT_POSITION}, Facing: {['N', 'E', 'S', 'W'][CURRENT_DIRECTION]} ---")
        
        log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, f"step_{step + 1}")
        
        print("   -> Resetting Yaw to ensure perfect alignment before new step...")
        attitude_handler.correct_yaw_to_target(movement_controller.chassis, get_compensated_target_yaw())
        
        perform_side_alignment_and_mapping(movement_controller, scanner, attitude_handler, occupancy_map, visualizer)

        # === PERFORM OBJECT DETECTION AFTER ALIGNMENT ===
        print("\n🎯 Performing object detection at current position...")
        perform_object_detection_scan(scan_duration=3.0)
        
        print("--- Performing Scan for Mapping (Front ToF Only) ---")
        is_front_occupied = scanner.get_front_tof_cm() < scanner.tof_wall_threshold_cm
        dir_map_abs_char = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
        occupancy_map.update_wall(r, c, dir_map_abs_char[CURRENT_DIRECTION], is_front_occupied, 'tof')
        
        occupancy_map.update_node(r, c, False, 'tof')
        visited_cells.add((r, c))
        visualizer.update_plot(occupancy_map, CURRENT_POSITION)
        
        nodes_visited = len(visited_cells)
        if nodes_visited >= IMU_COMPENSATION_START_NODE_COUNT:
            compensation_intervals = nodes_visited // IMU_COMPENSATION_NODE_INTERVAL
            new_compensation = compensation_intervals * IMU_COMPENSATION_DEG_PER_INTERVAL
            if new_compensation != IMU_DRIFT_COMPENSATION_DEG:
                IMU_DRIFT_COMPENSATION_DEG = new_compensation
                print(f"🔩 IMU Drift Compensation Updated: Visited {nodes_visited} nodes. New offset is {IMU_DRIFT_COMPENSATION_DEG:.1f}°")

        priority_dirs = [(CURRENT_DIRECTION + 1) % 4, CURRENT_DIRECTION, (CURRENT_DIRECTION - 1 + 4) % 4]
        moved = False
        dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        for target_dir in priority_dirs:
            target_r, target_c = r + dir_vectors[target_dir][0], c + dir_vectors[target_dir][1]
            
            if occupancy_map.is_path_clear(r, c, target_r, target_c) and (target_r, target_c) not in visited_cells:
                print(f"Path to {['N','E','S','W'][target_dir]} at ({target_r},{target_c}) seems clear. Attempting move.")
                movement_controller.rotate_to_direction(target_dir, attitude_handler)
                
                # === PERFORM OBJECT DETECTION AFTER ROTATION ===
                print("\n🎯 Performing object detection after rotation...")
                perform_object_detection_scan(scan_duration=2.0)
                
                print("    Ensuring gimbal is centered before ToF confirmation...")
                scanner.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed();
                time.sleep(0.1)
                
                print("    Confirming path forward with ToF...")
                is_blocked = scanner.get_front_tof_cm() < scanner.tof_wall_threshold_cm
                
                occupancy_map.update_wall(r, c, dir_map_abs_char[CURRENT_DIRECTION], is_blocked, 'tof')
                print(f"    ToF confirmation: Wall belief updated. Path is {'BLOCKED' if is_blocked else 'CLEAR'}.")
                visualizer.update_plot(occupancy_map, CURRENT_POSITION)

                if occupancy_map.is_path_clear(r, c, target_r, target_c):
                    axis_to_monitor = 'x' if ROBOT_FACE % 2 != 0 else 'y'
                    movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)

                    movement_controller.center_in_node_with_tof(scanner, attitude_handler)

                    CURRENT_POSITION = (target_r, target_c)
                    log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, "moved_to_new_node")
                    moved = True
                    break
                else:
                    print(f"    Confirmation failed. Path to {['N','E','S','W'][target_dir]} is blocked. Re-evaluating.")
        
        if not moved:
            print("No immediate unvisited path. Initiating backtrack...")
            backtrack_path = find_nearest_unvisited_path(occupancy_map, CURRENT_POSITION, visited_cells)

            if backtrack_path and len(backtrack_path) > 1:
                execute_path(backtrack_path, movement_controller, attitude_handler, scanner, visualizer, occupancy_map)
                print("Backtrack to new area complete. Resuming exploration.")
                continue
            else:
                print("🎉 EXPLORATION COMPLETE! No reachable unvisited cells remain.")
                break
    
    print("\n🎉 === EXPLORATION PHASE FINISHED ===")

# =============================================================================
# ===== MAIN EXECUTION BLOCK ==================================================
# =============================================================================
if __name__ == '__main__':
    ep_robot = None
    occupancy_map = OccupancyGridMap(width=4, height=4)
    attitude_handler = AttitudeHandler()
    movement_controller = None
    scanner = None
    ep_chassis = None
    
    # Object detection components
    tracker = ObjectTracker(use_gpu=USE_GPU)
    roi_state = {"x": ROI_X0, "y": ROI_Y0, "w": ROI_W0, "h": ROI_H0}
    manager = RMConnection()
    
    try:
        visualizer = RealTimeVisualizer(grid_size=4, target_dest=TARGET_DESTINATION)
        
        # Start reconnector thread
        reconn = threading.Thread(target=reconnector_thread, args=(manager,), daemon=True)
        reconn.start()
        
        # Wait for connection
        print("Waiting for robot connection...")
        while not manager.connected.is_set():
            time.sleep(0.5)
        
        # Get robot reference
        ep_robot = manager.get_robot()
        ep_chassis = ep_robot.chassis
        ep_gimbal = ep_robot.gimbal
        ep_tof_sensor = ep_robot.sensor
        ep_sensor_adaptor = ep_robot.sensor_adaptor
        
        print("GIMBAL: Centering gimbal...")
        ep_gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
        
        scanner = EnvironmentScanner(ep_sensor_adaptor, ep_tof_sensor, ep_gimbal, ep_chassis)
        movement_controller = MovementController(ep_chassis)
        attitude_handler.start_monitoring(ep_chassis)
        
        # Start object detection threads
        def is_detecting(): return is_detecting_flag["v"]
        
        cap_t  = threading.Thread(target=capture_thread_func, args=(manager, frame_queue), daemon=True)
        proc_t = threading.Thread(target=processing_thread_func,
                                  args=(tracker, frame_queue, TARGET_SHAPE, TARGET_COLOR, roi_state, is_detecting),
                                  daemon=True)
        ctrl_t = threading.Thread(target=control_thread_func, args=(manager, roi_state, is_detecting), daemon=True)
        
        cap_t.start(); proc_t.start(); ctrl_t.start()
        
        print("\n✅ All systems initialized!")
        print(f"🎯 Target: {TARGET_COLOR} {TARGET_SHAPE}")
        print("🗺️  Starting exploration with integrated object detection...\n")
        
        # Start exploration
        explore_with_ogm(scanner, movement_controller, attitude_handler, occupancy_map, visualizer)
        
        print(f"\n\n--- NAVIGATION TO TARGET PHASE: From {CURRENT_POSITION} to {TARGET_DESTINATION} ---")
        
        if CURRENT_POSITION == TARGET_DESTINATION:
            print("🎉 Robot is already at the target destination!")
        else:
            path_to_target = find_path_bfs(occupancy_map, CURRENT_POSITION, TARGET_DESTINATION)
            if path_to_target and len(path_to_target) > 1:
                print(f"✅ Path found to target: {path_to_target}")
                execute_path(path_to_target, movement_controller, attitude_handler, scanner, visualizer, occupancy_map, path_name="Final Navigation")
                print(f"🎉🎉 Robot has arrived at the target destination: {TARGET_DESTINATION}!")
            else:
                print(f"⚠️ Could not find a path from {CURRENT_POSITION} to {TARGET_DESTINATION}.")
        
    except KeyboardInterrupt: 
        print("\n⚠️ User interrupted exploration.")
        print("💾 Saving data before exit...")
    except Exception as e: 
        print(f"\n⚌ An error occurred: {e}")
        traceback.print_exc()
        print("💾 Saving data before exit...")
    finally:
        # Stop all threads
        stop_event.set()
        
        # Close OpenCV windows
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        
        # Save data
        try:
            print("💾 Saving map and timestamp data...")
            
            # Save map
            final_map_data = {'nodes': []}
            for r in range(occupancy_map.height):
                for c in range(occupancy_map.width):
                    cell = occupancy_map.grid[r][c]
                    cell_data = {
                        "coordinate": {"row": r, "col": c},
                        "probability": round(cell.get_node_probability(), 3),
                        "is_occupied": cell.is_node_occupied(),
                        "walls": {
                            "north": cell.walls['N'].is_occupied(),
                            "south": cell.walls['S'].is_occupied(),
                            "east": cell.walls['E'].is_occupied(),
                            "west": cell.walls['W'].is_occupied()
                        },
                        "wall_probabilities": {
                            "north": round(cell.walls['N'].get_probability(), 3),
                            "south": round(cell.walls['S'].get_probability(), 3),
                            "east": round(cell.walls['E'].get_probability(), 3),
                            "west": round(cell.walls['W'].get_probability(), 3)
                        }
                    }
                    final_map_data["nodes"].append(cell_data)

            with open(f"{SAVE_PATH}\\Mapping_Top.json", "w") as f:
                json.dump(final_map_data, f, indent=2)
            print("✅ Final Hybrid Belief Map saved to Mapping_Top.json")
            
            # Save timestamps
            timestamp_data = {
                "session_info": {
                    "start_time": POSITION_LOG[0]["iso_timestamp"] if POSITION_LOG else "N/A",
                    "end_time": POSITION_LOG[-1]["iso_timestamp"] if POSITION_LOG else "N/A",
                    "total_positions_logged": len(POSITION_LOG),
                    "grid_size": f"{occupancy_map.height}x{occupancy_map.width}",
                    "target_destination": list(TARGET_DESTINATION)
                },
                "position_log": POSITION_LOG
            }
            
            with open(f"{SAVE_PATH}\\Robot_Position_Timestamps.json", "w") as f:
                json.dump(timestamp_data, f, indent=2)
            print("✅ Robot position timestamps saved to Robot_Position_Timestamps.json")
            
            # Save detected objects
            with detected_objects_lock:
                detected_data = {
                    "session_info": {
                        "target_shape": TARGET_SHAPE,
                        "target_color": TARGET_COLOR,
                        "total_detected": len(detected_objects_log),
                        "total_fired": sum(1 for obj in detected_objects_log if obj["fired"])
                    },
                    "detected_objects": detected_objects_log
                }
                
                with open(f"{SAVE_PATH}\\Detected_Objects.json", "w") as f:
                    json.dump(detected_data, f, indent=2)
                print("✅ Detected objects saved to Detected_Objects.json")
            
        except Exception as save_error:
            print(f"❌ Error saving data: {save_error}")
        
        # Cleanup
        if ep_robot:
            print("🔌 Cleaning up and closing connection...")
            try:
                if scanner: scanner.cleanup()
                if attitude_handler and attitude_handler.is_monitoring: attitude_handler.stop_monitoring(ep_chassis)
                if movement_controller: movement_controller.cleanup()
                manager.close()
                print("🔌 Connection closed.")
            except Exception as cleanup_error:
                print(f"⚠️ Error during cleanup: {cleanup_error}")
        
        print("... You can close the plot window now ...")
        plt.ioff()
        plt.show()
