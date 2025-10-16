import math as m
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


def animation(pos_path, ang_path):
    pos_data = genfromtxt(pos_path)
    ang_data = genfromtxt(ang_path)

    pos_data.shape # (num_frames, (x, y, z)*22kps)


    num_frames = pos_data.shape[0]
    num_kp = 22
    num_axes = 3
    p_data = pos_data.T
    p_data = p_data.reshape(num_kp, num_axes, -1)
    a_data = ang_data.T
    a_data = a_data.reshape(num_kp, num_axes, -1)

    p = np.copy(p_data)
    a = np.copy(a_data)


    def Rotx(theta):
        return np.matrix([[1, 0, 0],
                          [0, m.cos(theta), -m.sin(theta)],
                          [0, m.sin(theta), m.cos(theta)]])


    def Roty(theta):
        return np.matrix([[m.cos(theta), 0, m.sin(theta)],
                          [0, 1, 0],
                          [-m.sin(theta), 0, m.cos(theta)]])


    def Rotz(theta):
        return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                          [m.sin(theta), m.cos(theta), 0],
                          [0, 0, 1]])


    def eulers_2_rot_matrix(x):
        gamma_x = x[0];
        beta_y = x[1];
        alpha_z = x[2];
        return Rotz(alpha_z) * Roty(beta_y) * Rotx(gamma_x)

    # convert the data from relative coordinates to absolute coordinates
    def rel2abs(p, a, num_frames):
        skel = np.zeros((num_kp, num_axes, num_frames))
        for i in range(num_frames):
            """
            1 Waist (absolute)
            2 Spine
            3 Chest
            4 Neck
            5 Head
            6 Head tip
            7 Left collar
            8 Left upper arm 
            9 Left forearm
            10 Left hand
            11 Right collar
            12 Right upper arm 
            13 Right forearm
            14 Right hand
            15 Left upper leg 
            16 Left lower leg 
            17 Left foot 
            18 Left leg toes
            19 Right upper leg 
            20 Right lower leg 
            21 Right foot
            22 Right leg toes
            """

            joint = p[:,:,i]
            joint_ang = a[:,:,i]

            # chest, neck, head
            rot_1 = eulers_2_rot_matrix(joint_ang[0,:]*np.pi/180);
            joint[1,:] =  rot_1@joint[1,:] + joint[0,:]
            rot_2 = rot_1*eulers_2_rot_matrix(joint_ang[1,:]*np.pi/180)
            joint[2,:] =  rot_2@joint[2,:] +  joint[1,:]
            rot_3 = rot_2*eulers_2_rot_matrix(joint_ang[2,:]*np.pi/180)
            joint[3,:] =  rot_3@joint[3,:] +  joint[2,:]
            rot_4 = rot_3*eulers_2_rot_matrix(joint_ang[3,:]*np.pi/180)
            joint[4,:] =  rot_4@joint[4,:] +  joint[3,:]
            rot_5 = rot_4*eulers_2_rot_matrix(joint_ang[4,:]*np.pi/180)
            joint[5,:] =  rot_5@joint[5,:] +  joint[4,:]

            # left-arm
            rot_6 = eulers_2_rot_matrix(joint_ang[2,:]*np.pi/180)
            joint[6,:] =  rot_6@joint[6,:] +  joint[2,:]
            rot_7 = rot_6*eulers_2_rot_matrix(joint_ang[6,:]*np.pi/180)
            joint[7,:] =  rot_7@joint[7,:] +  joint[6,:]
            rot_8 = rot_7*eulers_2_rot_matrix(joint_ang[7,:]*np.pi/180)
            joint[8,:] = rot_8@joint[8,:] +  joint[7,:]
            rot_9 = rot_8*eulers_2_rot_matrix(joint_ang[8,:]*np.pi/180)
            joint[9,:] = rot_9@joint[9,:] +  joint[8,:]

            # right-arm
            rot_10 = eulers_2_rot_matrix(joint_ang[2,:]*np.pi/180)
            joint[10,:] =  rot_10@joint[10,:] +  joint[2,:]
            rot_11 = rot_10*eulers_2_rot_matrix(joint_ang[10,:]*np.pi/180)
            joint[11,:] =  rot_11@joint[11,:] +  joint[10,:]
            rot_12 = rot_11*eulers_2_rot_matrix(joint_ang[11,:]*np.pi/180)
            joint[12,:] = rot_12@joint[12,:] +  joint[11,:]
            rot_13 = rot_12*eulers_2_rot_matrix(joint_ang[12,:]*np.pi/180)
            joint[13,:] = rot_13@joint[13,:] +  joint[12,:]

            # left-leg
            rot_14 = eulers_2_rot_matrix(joint_ang[0,:]*np.pi/180)
            joint[14,:] =  rot_14@joint[14,:] +  joint[0,:]
            rot_15 = rot_14*eulers_2_rot_matrix(joint_ang[14,:]*np.pi/180)
            joint[15,:] =  rot_15@joint[15,:] +  joint[14,:]
            rot_16 = rot_15*eulers_2_rot_matrix(joint_ang[15,:]*np.pi/180)
            joint[16,:] = rot_16@joint[16,:] +  joint[15,:]
            rot_17 = rot_16*eulers_2_rot_matrix(joint_ang[16,:]*np.pi/180)
            joint[17,:] = rot_17@joint[17,:] +  joint[16,:]

            # right-leg
            rot_18 = eulers_2_rot_matrix(joint_ang[0,:]*np.pi/180)
            joint[18,:] =  rot_18@joint[18,:] +  joint[0,:]
            rot_19 = rot_18*eulers_2_rot_matrix(joint_ang[18,:]*np.pi/180)
            joint[19,:] =  rot_19@joint[19,:] +  joint[18,:]
            rot_20 = rot_19*eulers_2_rot_matrix(joint_ang[19,:]*np.pi/180)
            joint[20,:] = rot_20@joint[20,:] +  joint[19,:]
            rot_21 = rot_20*eulers_2_rot_matrix(joint_ang[20,:]*np.pi/180)
            joint[21,:] = rot_21@joint[21,:] +  joint[20,:]

            skel[:,:,i] = joint
        return skel

    skel = rel2abs(p, a, num_frames)

    # find maximum and minimum values (for plotting)
    a = p_data.transpose(2,0,1).reshape(-1, 3)
    max_x = max(a[:, 0]) # 27
    min_x = min(a[:, 0]) #-26
    max_y = max(a[:, 1])
    min_y = min(a[:, 1])
    max_z = max(a[:, 2])
    min_z = min(a[:, 2])

    # order of joint connections
    J = np.array([[3, 5, 4, 2, 1, 2, 6, 7, 8, 2, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20],
                  [2, 4, 2, 1, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]])

    # Visualization

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)


    def get_plot(i):
        ax.cla()

        ax.set_title('2D plot using transformed data')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        joint = skel[:, :, i]

        for j in range(J.shape[1]):
            p1 = joint[J[0, j], :]
            p2 = joint[J[1, j], :]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'o-')


    plt.rcParams['animation.html'] = 'html5'
    anim = FuncAnimation(fig, get_plot, blit=False, frames=num_frames)
    return anim
    # plt.close()

    # anim


def angle_calc(joint1, joint2, joint3):
    a = np.array([joint1[0], joint1[1], joint1[2]])  # First
    b = np.array([joint2[0], joint2[1], joint2[2]])  # Mid
    c = np.array([joint3[0], joint3[1], joint3[2]])  # End

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return round(np.degrees(angle), 2)


if __name__ == "__main__":
    position = "data/ui-prmd/Correct/Kinect/Positions/m10_s02_positions.txt"
    angle = "data/ui-prmd/Correct/Kinect/Angles/m10_s02_angles.txt"
    anim = animation(position, angle)

    angle_data = genfromtxt(angle)

    plt.plot(angle_data[:, 33])
    plt.plot(angle_data[:, 34])
    plt.plot(angle_data[:, 35])
    plt.plot(angle_data[:, 36])
    plt.plot(angle_data[:, 37])
    plt.plot(angle_data[:, 38])

    print(f"Number of frames, Joints (X,Y,Z * 22 Joints): {pos_data.shape}") # (num_frames, (x, y, z)*22kps)

num_frames = pos_data.shape[0]
num_frames = 22

joints_list = ["Waist" ,"Spine", "Chest", "Neck", "Head", "Head tip", "Left collar", "Left upper arm", "Left forearm",
               "Left hand", "Right collar", "Right upper arm", "Right forearm", "Right hand", "Left upper leg",
               "Left lower leg", "Left foot", "Left leg toes", "Right upper leg", "Right lower leg", "Right foot",
               "Right leg toes"]

left_angles = []
right_angles = []
for frame in pos_data:
    left_collar = frame[18:21]
    left_upper_arm = frame[21:24]
    left_forearm = frame[24:27]
    # left_hand = frame[27:30]
    left_angles.append(angle_calc(left_collar, left_upper_arm, left_forearm))
    # left_angles.append(angle_calc(left_collar, left_upper_arm, left_hand))


    right_collar = frame[30:33]
    right_upper_arm = frame[33:36] + right_collar
    right_forearm = frame[36:39] + right_upper_arm
    # right_hand = frame[39:42]
    right_angles.append(angle_calc(right_collar, right_upper_arm, right_forearm))
    # right_angles.append(angle_calc(right_collar, right_upper_arm, right_hand))

plt.plot(right_angles)
plt.plot(left_angles)


angle_path = "data/ui-prmd/Correct/Kinect/Angles/m07_s01_angles.txt"
ang_data = genfromtxt(angle_path)

plt.plot(ang_data[:,33])
plt.plot(ang_data[:,34])
plt.plot(ang_data[:,35])

