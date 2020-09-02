#!/usr/bin/env python
# Copyright (c) 2017 Zi Wang
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import kitchen2d.kitchen_stuff as ks
from kitchen2d.kitchen_stuff import Kitchen2D
from kitchen2d.gripper import Gripper
import numpy as np

settings = {
    0: {
        'do_gui': False,
        'sink_w': 4.,
        'sink_h': 5.,
        'sink_d': 1.,
        'sink_pos_x': 50,
        'left_table_width': 100.,
        'right_table_width': 100.,
        'faucet_h': 8.,
        'faucet_w': 5.,
        'faucet_d': 0.5,
        'planning': False,
        'save_fig': False
    },
    1: {
        'do_gui': False,
        'sink_w': 24.,
        'sink_h': 20.,
        'sink_d': 0.5,
        'sink_pos_x': -12.,
        'left_table_width': 100.,
        'right_table_width': 100.,
        'faucet_h': 15.,
        'faucet_w': 3.,
        'faucet_d': 0.5,
        'planning': False,
        'save_fig': True
    }
}

def get_water_filter(ims):
    filter = (ims.take(0, axis=-1) < 230) * \
             (ims.take(1, axis=-1) < 230) * \
             (ims.take(2, axis=-1) > 200)
    return filter[..., np.newaxis] * ims

def compute_metric(curr_im, goal_im, metric_type):
    curr_im = curr_im.astype(np.float32)
    goal_im = goal_im.astype(np.float32)
    if metric_type == 'pixel_l2':
        num_dims = len(curr_im.shape)
        curr_im = get_water_filter(curr_im)
        goal_im = get_water_filter(goal_im)
        return (((curr_im - goal_im) /
                 255.)**2).mean(axis=(num_dims - 3, num_dims - 2, num_dims - 1))
    elif metric_type == 'pixel_l2_goal_overlap':
        num_dims = len(curr_im.shape)
        curr_im = get_water_filter(curr_im)
        goal_im = get_water_filter(goal_im)
        return (((curr_im - goal_im) * (goal_im > 0) /
                 255.)**2).mean(axis=(num_dims - 3, num_dims - 2, num_dims - 1))


class MultiCups(object):
    def __init__(self, version='v2'):
        self.version = version
        #grasp_ratio, relative_pos_x, relative_pos_y, dangle, cw1, ch1, cw2, ch2
        self.x_range = np.array(
            [[0., -10., 1., -np.pi/3, 3., 3., 4., 4., 3., 3., -10., -10., 2.],
            [1., 10., 10., np.pi/3, 5., 4., 4., 5., 4., 4., -2., 10., 10.]])
        if version == 'v1':
            self.maxspeed = 0.1
            # self.gravity = 10.0
            self.base_cup_type = 'cup'
            self.water_color = (26, 130, 252)
            self.interpolation = 'INTER_LANCZOS4'
        elif version == 'v2':
            self.x_range[0, 2] = 5
            # self.gravity = 1.0
            self.maxspeed = 0.2
            self.base_cup_type = 'cup3'
            self.water_color = (0, 0, 255)
            self.interpolation = 'INTER_AREA'
        else:
            raise NotImplementedError
            #[1., 10., 10., np.pi, 8., 5., 4.5, 5.]]) this is the upper bound used in the paper.
        self.lengthscale_bound = np.array([np.ones(8)*0.1, [0.15, 0.5, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5]])
        self.context_idx = [4, 5, 6, 7]
        self.param_idx = [0, 1, 2, 3]
        self.dx = len(self.x_range[0])
        self.task_lengthscale = np.ones(8)*10
        self.do_gui = True
    def check_legal(self, x):
        grasp_ratio, rel_x, rel_y, dangle, \
        cw1, ch1, cw2, ch2, cw3, ch3, \
        pos_x1, pos_x2, pos_x3 = x
        dangle *= np.sign(rel_x)
        settings[0]['do_gui'] = self.do_gui
        kitchen = Kitchen2D(**settings[0])
        kitchen.gui_world.colors['water'] = self.water_color
        gripper = Gripper(kitchen, (5,8), 0)
        cup1 = ks.make_cup(kitchen, (pos_x1, 0), 0, cw1, ch1, 0.5, user_data=self.base_cup_type)
        cup2 = ks.make_cup(kitchen, (pos_x2, 0), 0, cw2, ch2, 0.5, user_data='cup2')
        cup3 = ks.make_cup(kitchen, (pos_x3, 0), 0, cw3, ch3, 0.5, user_data=self.base_cup_type)
        gripper.set_grasped(cup2, grasp_ratio, (pos_x2, 0), 0)
        gripper.set_position((rel_x, rel_y), 0)
        if not kitchen.planning:
            g2 = gripper.simulate_itself()
            _, collision = g2.check_path_collision((rel_x, rel_y), 0, (rel_x, rel_y), dangle)
            if collision:
                return False
        self.kitchen = kitchen
        self.gripper = gripper
        self.cup1 = cup1
        self.cup2 = cup2
        self.cup3 = cup3
        self.properties = x[4:10]
        return True

    def sampled_x(self, n, n_freeze_poses = 100):
        i = 0
        while i < n:
            x = np.random.uniform(self.x_range[0], self.x_range[1])
            # Sizes
            x[4:10] = 4, 4, 4, 4, 4, 4
            # Grasp ratio
            x[0] = 0.5
            # Height of pouring cup
            x[2] += 2.
            # Angle of pouring cup
            x[3] = np.random.choice([-np.pi/3, np.pi/3])
            # Positions
            if i % n_freeze_poses == 0:
                tmp_x_poses = (x[10], x[12])
            x[10] = tmp_x_poses[0]
            x[12] = tmp_x_poses[1]

            legal = self.check_legal(x)
            if legal:
                i += 1
                yield x

    def sample_same_context(self, x):
        # Change angle
        x[3] = np.random.choice([-np.pi/3, np.pi/3])
        # Change y position
        x[1] = np.random.uniform(self.x_range[0, 1], self.x_range[1, 1])
        return x

    def step(self, action):
        traj = self.gripper.apply_lowlevel_control(
          self.gripper.position + action[:2],
          self.gripper.angle + action[2],
          maxspeed=self.maxspeed)
        return traj

    def setup(self, x, liquid_state='initial', image_name=None):
        '''
        x = (grasp_ratio, rel_x, rel_y, dangle, cw1, ch1, cw2, ch2)
        (rel_x, rel_y): relative position
        dangle: relative angle
        cw1, ch1: width and height for cup 1
        cw2, ch2: width and height for cup 2
        '''
        while not self.check_legal(x):
            x = self.sample_same_context(x)
        if not self.check_legal(x):
            return -1.
        # grasp_ratio, rel_x, rel_y, dangle = x[:4]
        # dangle *= np.sign(rel_x)
        # if self.kitchen.planning:
        #     self.gripper.close()
        #     dpos = self.cup1.position + (rel_x, rel_y)
        #     self.gripper.set_position(dpos, dangle)
        #     self.kitchen.image_name = image_name
        #     self.kitchen.step()
        #     return
        if liquid_state == 'initial':
            self.kitchen.gen_liquid_in_cup(self.cup2, 500)
        elif liquid_state == 'all_in_cup3':
            self.kitchen.gen_liquid_in_cup(self.cup3, 500)
            assert self.get_num_particles()[1] == 500
        elif liquid_state == 'all_in_cup1':
            self.kitchen.gen_liquid_in_cup(self.cup1, 500)
            assert self.get_num_particles()[0] == 500
        elif liquid_state == 'half_in_cup1':
            self.kitchen.gen_liquid_in_cup(self.cup2, 250)
            self.kitchen.gen_liquid_in_cup(self.cup1, 250)
            assert self.get_num_particles()[0] == 250
        elif liquid_state == 'half_in_cup3':
            self.kitchen.gen_liquid_in_cup(self.cup3, 250)
            self.kitchen.gen_liquid_in_cup(self.cup2, 250)
            assert self.get_num_particles()[1] == 250
        elif liquid_state == 'goal':
            self.kitchen.gen_liquid_in_cup(self.cup3, 250)
            self.kitchen.gen_liquid_in_cup(self.cup1, 250)
            assert self.get_num_particles()[0] == 250
            assert self.get_num_particles()[1] == 250
        else:
            raise NotImplementedError

        self.gripper.compute_post_grasp_mass()
        self.gripper.close(timeout=0.1)
        self.gripper.check_grasp(self.cup2)
        self.step((0, 0, x[3]))
        # To flat the water
        self.step((0, 0, -x[3]/4))
        self.step((0, 0, x[3]/4))
        self.step((0, 0, -x[3]/16))
        self.step((0, 0, x[3]/16))
        self.x = x.copy()

    def get_state(self):
      return self.get_num_particles()
      # self.gripper.compute_post_grasp_mass()
      # return np.array([*self.gripper.position,
      #                  self.gripper.angle,
      #                  self.gripper.mass,
      #                  *self.cup1.position,
      #                  *self.cup2.position,
      #                  *self.cup3.position,
      #                  *self.properties
      #                  ])

    def render(self):
        return self.kitchen.render(self.interpolation)

    def save_observation(self, filepath):
        return self.kitchen.save_observation(filepath)

    def reset(self, x=None, liquid_state='initial'):
        if x is None:
            x = list(self.sampled_x(1))[0]
        self.setup(x, liquid_state)
        return x
        # return self.render()

    def get_num_particles(self):
        in_cup1 = 0
        in_cup3 = 0
        for particle in self.kitchen.liquid.particles:
          if (np.abs(self.cup1.position - particle.position) < (2,3)).all():
            in_cup1 += 1
          elif (np.abs(self.cup3.position - particle.position) < (2,3)).all():
            in_cup3 += 1
        return np.array([in_cup1, in_cup3])


class MultiCupsFaucet(MultiCups):
    def __init__(self, version='v2'):
        super(MultiCupsFaucet, self).__init__(version=version)
        self.x_range = np.array(
            [[0., -10., 1., -np.pi/3, 3., 3., 4., 4., 3., 3., -10., -10., -10.],
            [1., 10., 10., np.pi/3, 5., 4., 4., 5., 4., 4., 0., 10., 0.]])

    def check_legal(self, x):
        grasp_ratio, rel_x, rel_y, dangle, \
        cw1, ch1, cw2, ch2, cw3, ch3, \
        pos_x1, pos_x2, pos_x3 = x
        dangle *= np.sign(rel_x)
        settings[1]['do_gui'] = self.do_gui
        kitchen = Kitchen2D(**settings[1])
        kitchen.gui_world.colors['water'] = self.water_color
        gripper = Gripper(kitchen, (5,8), 0)
        cup1 = ks.make_static_cup(kitchen, (pos_x1, 0), 0, cw1, ch1, 0.5, user_data=self.base_cup_type)
        cup2 = ks.make_cup(kitchen, (pos_x2, 0), 0, cw2, ch2, 0.5, user_data='cup2')
        cup3 = ks.make_static_cup(kitchen, (pos_x3, 0), 0, cw3, ch3, 0.5, user_data=self.base_cup_type)
        kitchen.add_obstacles(obstacles=[((-13.5, 0), (0.5,20)),
                                         ((13.5, 0), (0.5,20)),
                                         ((0, 21), (26,0.5))])
        gripper.set_grasped(cup2, grasp_ratio, (pos_x2, 0), 0)
        gripper.set_position((rel_x, rel_y), 0)
        if not kitchen.planning:
            g2 = gripper.simulate_itself()
            _, collision = g2.check_path_collision((rel_x, rel_y), 0, (rel_x, rel_y), dangle)
            if collision:
                return False
        self.kitchen = kitchen
        self.gripper = gripper
        self.cup1 = cup1
        self.cup2 = cup2
        self.cup3 = cup3
        self.properties = x[4:10]
        return True

    def sampled_x(self, n, n_freeze_poses = 100):
        i = 0
        while i < n:
            x = np.random.uniform(self.x_range[0], self.x_range[1])
            # Sizes
            x[4:10] = 4, 4, 4, 4, 4, 4
            # Grasp ratio
            x[0] = 0.5
            # Height of pouring cup
            x[2] += 2.
            # Angle of pouring cup
            x[3] = np.random.choice([-np.pi/3, np.pi/3])
            # Positions
            if i % n_freeze_poses == 0:
                tmp_x_poses = (x[10], x[12])
            # Trick for sampling two numbers that are at least 4 apart.
            tmp_x_poses = sorted(tmp_x_poses)
            x[10] = tmp_x_poses[0]
            x[12] = tmp_x_poses[1] + 4

            legal = self.check_legal(x)
            if legal:
                i += 1
                yield x

    
