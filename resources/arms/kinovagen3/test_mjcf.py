import math
import mujoco
import mujoco_viewer
import os
import numpy as np
import KinovaGen3

class TestArm:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(f"./resources/arms/kinovagen3/mjcf/kinova_hammer_isaacsim_pull.xml")
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

    def step(self,ctrl):
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)

    def render(self):
        if self.viewer.is_alive:
            self.viewer.render()

    def show_qpos(self):
        print(self.data.qpos)

    def get_dof_pos(self):
        dof_pos = self.data.qpos[:7]
        return dof_pos
    
    def get_dof_vel(self):
        dof_vel = self.data.qvel[:7]
        return dof_vel
    
    def get_sensor_hand_pos(self):
        hand_pos = self.data.sensor(f"hand_pos").data
        print("sensor_hand_pos: ", hand_pos)
        return hand_pos
    
    def get_sensor_hand_quat(self):
        hand_quat = self.data.sensor(f"hand_quat").data
        print("sensor_hand_quat: ", hand_quat)
        return hand_quat
    
    def get_sensor_hand_vel(self):
        hand_vel = self.data.sensor(f"hand_velo").data
        print("sensor_hand_vel: ",hand_vel)
        return hand_vel
    
    def get_sensor_hand_acc(self):
        hand_acc = self.data.sensor(f"hand_acc").data
        print("hand_acc: ",hand_acc)
        return hand_acc
    
    def get_sensor_nail_pos(self):
        nail_pos = self.data.sensor(f"nail_pos").data
        print("sensor_nail_pos: ",nail_pos)
        return nail_pos
    
    def get_sensor_hammer_head_force(self):
        hf = self.data.sensor(f"hammer_head_force").data
        print("sensor_hammer_head_force: ".hf)
        return hf
    
    def direct_dof_pos_control(self,joint_ang):
        self.data.qpos[:7] = joint_ang
        ctrl = np.zeros(7)
        self.step(ctrl)
        self.render()
    
    def direct_dof_vel_control(self,joint_vel):
        self.data.qvel[:7] = joint_vel
        ctrl = np.zeros(7)
        self.step(ctrl)
        self.render()

    def ik_tip_vel_control(self,tip_vel):
        current_q = self.get_dof_pos()
        jv = KinovaGen3.multicriteria_ik_damped(current_q,tip_vel)
        self.direct_dof_vel_control(jv)

    def quaternion_2_euler(self,q):
        w,x,y,z = q
        eps = 0.0009765625
        thres = 0.5 - eps

        test = w * y - x * z
        
        if test < -thres or test > thres:
            sign = 1 if test > 0 else -1
            gamma = -2 * sign * math.atan2(x, w)
            beta = sign * (math.pi / 2)
            alpha = 0
        else:
            alpha = math.atan2(2 * (y*z + w*x), w*w - x*x - y*y + z*z)
            beta = math.asin(-2 * (x*z - w*y))
            gamma = math.atan2(2 * (x*y + w*z), w*w + x*x - y*y - z*z)
        return alpha,beta,gamma
    
    def _is_rotation_matrix(self,R):
        Rt = np.transpose(R)
        should_be_identity = np.dot(Rt,R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - should_be_identity)
        return n < 1e-6
    
    def _rotation_matrix_2_eular_angles(self,R):
        assert(self._is_rotation_matrix(R))
        sy = np.sqrt(R[0,0]*R[0,0]+R[1,0]*R[1,0])
        singular = sy<1e-6
        if not singular:
            x = np.arctan2(R[2,1],R[2,2])
            y = np.arctan2(-R[2,0],sy)
            z = np.arctan2(R[1,0],R[0,0])
        else:
            x = np.arctan2(-R[1,2],R[1,1])
            y = np.arctan2(-R[2,0],sy)
            z = 0
        return np.array([x,y,z])

    def simulate(self):
        home_angles_arm = [0, 0.4, np.pi, -np.pi+1.4, 0, -1, np.pi/2]
        for i in range(500):
            self.direct_dof_pos_control(home_angles_arm)
        print("dof_pos",self.get_dof_pos())
        for i in range(400):
            self.ik_tip_vel_control([0.1,0.1,0.2,0,0,0])
        print("dof_pos: ", self.get_dof_pos())
        print("sensor hand pos: ", self.get_sensor_hand_pos())
        hand_quat = self.get_sensor_hand_quat()
        print("sensor hand quat: ", hand_quat)
        print("sensor hand euler: ", self.quaternion_2_euler(hand_quat))
        fk_end_effector_pos = KinovaGen3.forward_kinematics(self.get_dof_pos())
        print("fk end effector position: ", fk_end_effector_pos[0])
        print("fk end effector position eular: ", self._rotation_matrix_2_eular_angles(fk_end_effector_pos[1]))
        print("jacobian:",KinovaGen3.jacobian(home_angles_arm))

if __name__ == "__main__":
    gen3 = TestArm()
    gen3.simulate()