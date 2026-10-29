"""UR RTDE connection. `rtde_control` / `rtde_receive` are imported only when connecting."""

import time

import numpy as np

from kossmc_issf.console import debug, info

# getRobotMode() codes, from the ur_rtde documentation (rtde_receive_interface.h)
ROBOT_MODES = {
    -1: "NO_CONTROLLER", 0: "DISCONNECTED", 1: "CONFIRM_SAFETY", 2: "BOOTING", 3: "POWER_OFF",
    4: "POWER_ON", 5: "IDLE", 6: "BACKDRIVE", 7: "RUNNING", 8: "UPDATING_FIRMWARE",
}
# getSafetyMode() codes, from the Universal Robots Client Library (ur/datatypes.h, SafetyMode)
SAFETY_MODES = {
    1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY", 5: "SAFEGUARD_STOP",
    6: "SYSTEM_EMERGENCY_STOP", 7: "ROBOT_EMERGENCY_STOP", 8: "VIOLATION", 9: "FAULT",
    10: "VALIDATE_JOINT_ID", 11: "UNDEFINED_SAFETY_MODE", 12: "AUTOMATIC_MODE_SAFEGUARD_STOP",
    13: "SYSTEM_THREE_POSITION_ENABLING_STOP", 14: "TP_THREE_POSITION_ENABLING_STOP",
    15: "IMMI_EMERGENCY_STOP", 16: "IMMI_SAFEGUARD_STOP", 17: "PROFISAFE_WAITING_FOR_PARAMETERS",
    18: "PROFISAFE_AUTOMATIC_MODE_SAFEGUARD_STOP", 19: "PROFISAFE_SAFEGUARD_STOP",
    20: "PROFISAFE_EMERGENCY_STOP", 22: "SAFETY_API_SAFEGUARD_STOP",
}
ROBOT_MODE_RUNNING = 7
SAFETY_MODE_NORMAL = 1


def mode_name(table, code):
    """"RUNNING (7)"; "UNKNOWN (n)" for a code that is not in the table."""
    return f"{table.get(code, 'UNKNOWN')} ({code})"


def report_status(robot_mode, safety_mode):
    """Connection line, plus a warning if the robot is not running in normal safety mode.

    The run is not stopped by the warning.
    """
    info(f"[RTDE] Connected | robot {mode_name(ROBOT_MODES, robot_mode)} | "
         f"safety {mode_name(SAFETY_MODES, safety_mode)}")
    if robot_mode != ROBOT_MODE_RUNNING or safety_mode != SAFETY_MODE_NORMAL:
        info("[WARN] robot not ready: check PolyScope (power on, brakes released, no protective stop)")


class RTDERobot:
    """Thin wrapper over RTDEControlInterface / RTDEReceiveInterface."""

    def __init__(self, ip):
        self.ip = ip
        self.rtde_c = None
        self.rtde_r = None

    def connect(self):
        """Connect and print the robot status. Returns False (after printing the error) on failure."""
        import rtde_control
        import rtde_receive

        debug("[RTDE] Connecting...")
        try:
            self.rtde_c = rtde_control.RTDEControlInterface(self.ip)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ip)
            robot_mode, safety_mode = self.rtde_r.getRobotMode(), self.rtde_r.getSafetyMode()
            debug("[RTDE] Connected!\n")
            debug("Robot mode :", robot_mode)
            debug("Safety mode:", safety_mode)
            debug("isConnected :", self.rtde_c.isConnected())
            debug("isProgramRunning :", self.rtde_c.isProgramRunning())
        except Exception as e:
            print(f"[ERROR] {e}"); return False
        report_status(robot_mode, safety_mode)
        return True

    def get_q(self):
        return np.array(self.rtde_r.getActualQ())

    def get_qd(self):
        return np.array(self.rtde_r.getActualQd())

    def get_tcp_pose(self):
        return self.rtde_r.getActualTCPPose()

    def read_state(self):
        """State of one control cycle: (q, TCP pose, qd), read in this order."""
        q_actual_rtde = np.array(self.rtde_r.getActualQ())
        tcp_pose = self.rtde_r.getActualTCPPose()
        qd_actual = np.array(self.rtde_r.getActualQd())
        return q_actual_rtde, tcp_pose, qd_actual

    def get_robot_mode(self):
        return self.rtde_r.getRobotMode()

    def get_safety_mode(self):
        return self.rtde_r.getSafetyMode()

    def move_j(self, target, speed, accel):
        self.rtde_c.moveJ(target, speed, accel, False)

    def speed_j(self, dq_cmd):
        self.rtde_c.speedJ(dq_cmd.tolist(), acceleration= 1, time=0.008)

    def stop(self):
        self.rtde_c.speedStop()
        self.rtde_c.stopScript()

    def disconnect(self):
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()


def move_init(robot, twin, target, mode="joint", speed=0.5, accel=0.5, sleep=time.sleep):
    """moveJ to `target`, then sync the PyBullet twin. Raises if the joint error is > 0.05 rad."""
    if mode == "joint":
        debug("BEFORE moveJ")
        debug("q actual =", np.round(robot.get_q(),3))

        robot.move_j(target, speed, accel)

        sleep(1)

        debug("AFTER moveJ")
        debug("q actual =", np.round(robot.get_q(),3))

    # sync the twin after the move
    q_real = robot.get_q()
    dq_real = robot.get_qd()
    twin.reset_state(q_real, dq_real)
    err = np.linalg.norm(q_real - np.array(target))

    if err > 0.05:
        raise RuntimeError(
            f"Robot did not reach q_init. Error = {err:.3f} rad"
        )

    debug(f"[move_init] dist to target = {np.linalg.norm(q_real - np.array(target)):.6f} rad")
