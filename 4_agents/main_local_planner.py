import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import bernstein_coeff_order10_arbitinterval

import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy
import time
from jax import vmap, random
import json
import mpc_expert_local_sample
from collision_utils import compute_no_collisions
import way_points



class planning_traj():

    def __init__(self):

        self.t_fin = 10 #### time horizon
        self.num = 100 #### number of steps in prediction
        self.tot_time = np.linspace(0, self.t_fin, self.num)
        self.num_batch = 1
        self.maxiter_proj = 100
        self.maxiter_cem = 1
        self.maxiter_mpc = 2000
        self.num_sample = 10
        self.num_sample_exec = 6

        self.weight_smoothness = 0
        self.rho_obs = 1
        self.rho_ineq = 1
        self.way_point_shape = 0

        self.initial_trajectories = []

        self.x_agent = []
        self.y_agent = []
        self.z_agent = []

        self.x_init = []
        self.y_init = []
        self.z_init = []

        self.x_goal = []
        self.y_goal = []
        self.z_goal = []

        self.vx_init = []
        self.vy_init = []
        self.vz_init = []

        self.vx_goal = []
        self.vy_goal = []
        self.vz_goal = []

        self.ax_init = []
        self.ay_init = []
        self.az_init = []

        self.ax_goal = []
        self.ay_goal = []
        self.az_goal = []

        self.x_agent = []
        self.y_agent = []
        self.z_agent = []


        self.a_obs = 0
        self.b_obs = 0
        self.c_obs = 0
    
        self.vx_obs = []
        self.vy_obs = []
        self.vz_obs = []

        self.a_obs_agent = 0
        self.b_obs_agent = 0
        self.c_obs_agent = 0

        self.v_max = 0.7
        self.v_min = 0.02
        self.v_des = 0.4

        self.a_max = 1.1

        self.num_obs = 0
        self.num_obs_dy = 0

        self.t_update = 0.1
        self.waypoint_shape = 1000

    ##################################
    def data_prep(self, obstacles):    

        self.robot_config = obstacles["robot_config"]
        self.weight_smoothness = obstacles["weight_smoothness"]
        self.rho_obs = obstacles["rho_obs"]
        self.config_name = obstacles["name"]

        print ("Configuration: ",self.config_name)


        self.x_init_obs = jnp.hstack(obstacles["x_init_obs"])
        self.y_init_obs = jnp.hstack(obstacles["y_init_obs"])
        self.z_init_obs = jnp.hstack(obstacles["z_init_obs"])
        
        self.num_obs = len(self.x_init_obs)
        

        self.vx_obs = jnp.zeros((self.num_obs, self.num))
        self.vy_obs = jnp.zeros((self.num_obs, self.num))
        self.vz_obs = jnp.zeros((self.num_obs, self.num))

        with open(self.robot_config, 'r') as f:
            robot_data = json.load(f)
        self.n_agents = 4#len(robot_data["agents"])
        # self.n_agents = len(robot_data["agents"])
        self.num_obs_dy = self.n_agents

        self.a_obs_agent = 0.2 + 0.2 + 0.05#robot_data["agents"][0]["radius"] * 2 + 0.04
        self.b_obs_agent = 0.2 + 0.2 + 0.05#self.a_obs_agent  
        self.c_obs_agent = 0.2 + 0.2 + 0.05#self.a_obs_agent 

        self.obs_rad = obstacles["obst_rad"]
        self.a_obs = 0.2+ 0.2+ 0.05#2*self.obs_rad  #+ 0.5 * self.a_obs_agent
        self.b_obs = 0.2+ 0.2+ 0.05#2*self.obs_rad  #+ 0.5 * self.a_obs_agent
        self.c_obs = 1.4+ 0.2+ 0.05#2*self.obs_rad  #+ 0.5 * self.a_obs_agent
       
        trajectories = []

        self.x_waypoint =  np.load("/home/fatemeh/multi-agent/test_1/4_agents_3d/way_points/x_waypoint4.npy")
        self.y_waypoint =  np.load("/home/fatemeh/multi-agent/test_1/4_agents_3d/way_points/y_waypoint4.npy")
        self.z_waypoint =  np.load("/home/fatemeh/multi-agent/test_1/4_agents_3d/way_points/z_waypoint4.npy")

        self.x_init = self.x_waypoint[:,0] 
        self.y_init = self.y_waypoint[:,0]
        self.z_init = self.z_waypoint[:,0]

        self.x_init_robot = self.x_waypoint[:,0] 
        self.y_init_robot = self.y_waypoint[:,0]
        self.z_init_robot = self.z_waypoint[:,0]

        self.x_goal = self.x_waypoint[:,-1]
        self.y_goal = self.y_waypoint[:,-1]
        self.z_goal = self.z_waypoint[:,-1]

        self.x_goal_fin = self.x_waypoint[:,-1]
        self.y_goal_fin = self.y_waypoint[:,-1]
        self.z_goal_fin = self.z_waypoint[:,-1]

        self.x_agent = self.x_waypoint
        self.y_agent = self.y_waypoint
        self.z_agent = self.z_waypoint

        self.vx_init = 0.05 * jnp.ones(self.n_agents)
        self.vy_init = 0.0 * jnp.ones(self.n_agents)
        self.vz_init = 0.0 * jnp.ones(self.n_agents)
    
        self.vx_goal = 0.0 * jnp.ones(self.n_agents)
        self.vy_goal = 0.0 * jnp.ones(self.n_agents)
        self.vz_goal = 0.0 * jnp.ones(self.n_agents)

        self.ax_init = 0.0 * jnp.ones(self.n_agents)
        self.ay_init = 0.0 * jnp.ones(self.n_agents)
        self.az_init = 0.0 * jnp.ones(self.n_agents)
    
        self.ax_goal = 0.0 * jnp.ones(self.n_agents)
        self.ay_goal = 0.0 * jnp.ones(self.n_agents)
        self.az_goal = 0.0 * jnp.ones(self.n_agents)

        self.M0 = int(self.n_agents)
        self.Tmax = int(2*self.maxiter_mpc)
        self.active_ids = jnp.arange(self.M0)

        self.x_hist = jnp.full((self.M0, self.Tmax), jnp.nan)
        self.y_hist = jnp.full((self.M0, self.Tmax), jnp.nan)
        self.z_hist = jnp.full((self.M0, self.Tmax), jnp.nan)
        self.hist_t  = 0

        self.prev_n  = int(self.n_agents)

        self.removed_mask_all = jnp.zeros((self.M0,), dtype=bool) 
        self.obs_idx_for_removed = -jnp.ones((self.M0,), dtype=jnp.int32)  

        self.active_mask = jnp.ones((self.n_agents,), dtype=bool)

        x_dummy, y_dummy, z_dummy, x_waypoint_dummy, y_waypoint_dummy, z_waypoint_dummy = self.make_dummy_trajs()

        self.x_dummy = x_dummy
        self.y_dummy = y_dummy
        self.z_dummy = z_dummy

        self.x_waypoint_dummy = x_waypoint_dummy
        self.y_waypoint_dummy = y_waypoint_dummy
        self.z_waypoint_dummy = z_waypoint_dummy

        self.x_obs_dynamic = 200*jnp.ones((self.n_agents, self.num))[:,0]#x_dummy[:,0] + 100
        self.y_obs_dynamic = 200*jnp.ones((self.n_agents, self.num))[:,0]#y_dummy[:,0] + 10
        self.z_obs_dynamic = 0.0*jnp.ones((self.n_agents, self.num))[:,0]#z_dummy[:,0]

        self.num_obs_dynamic = self.n_agents

        self.vx_obs_dynamic = jnp.zeros((self.n_agents, self.num))
        self.vy_obs_dynamic = jnp.zeros((self.n_agents, self.num))
        self.vz_obs_dynamic = jnp.zeros((self.n_agents, self.num))


    #######################
    def interp_signal(self, signal):

        t_old = jnp.linspace(0, 1, 100)   # 100 points
        t_new = jnp.linspace(0, 1, 1000)   

        return jnp.interp(t_new, t_old, signal)

    ###########################################33
    def make_dummy_trajs(self,):

        t = jnp.linspace(0.0, 1.0, self.num)
        t2 = jnp.linspace(0.0, 1.0, 1000)

        starts = 100.0 + 10.0 * jnp.arange(self.n_agents)
        x_dummy = starts[:, None] + 1.0 * t[None, :]  
        y_dummy = jnp.full((self.n_agents, self.num), 100.0) + 0.1 * starts[:, None]
        z_dummy = jnp.full((self.n_agents, self.num), 100.0)

        starts = 100.0 + 10.0 * jnp.arange(self.n_agents)
        x_waypoint_dummy = starts[:, None] + 1.0 * t2[None, :] 
        y_waypoint_dummy = jnp.full((self.n_agents, 1000), 1000.0) + 0.1 * starts[:, None]
        z_waypoint_dummy = jnp.full((self.n_agents, 1000), 1000.0)

        # print(x_dummy[:, 0], x_dummy[:,-1], "dummy")
        # print(y_dummy[:, 0], y_dummy[:,-1], "dummy")
        # print(z_dummy[:, 0], z_dummy[:,-1], "dummy")

        return x_dummy, y_dummy, z_dummy, x_waypoint_dummy, y_waypoint_dummy, 0.0*z_waypoint_dummy

    ##################################
    def _unit(self, x0, xg, y0, yg, z0, zg):

        step = []
        move = []

        for i in range(np.shape(x0)[0]):

            dx = xg - x0[i]
            dy = yg - y0[i]
            dz = zg - z0[i]
            d  = jnp.sqrt(dx*dx + dy*dy + dz*dz) + 1e-9
            ux = dx / d
            uy = dy / d
            uz = dz / d

            steps_needed = jnp.ceil(d / (self.v_max * self.t_update))
            k = jnp.arange(self.num)[None, :]                                       
            moving = (k < steps_needed[:, None])

        step.append(steps_needed)
        move.append(moving)

        step = np.array(step)
        move = np.array(move)

        return ux, uy, uz, d, steps_needed, moving

    ############################################

    def planner(self, ):

        ############# Setting 
        initial_state_x = jnp.vstack(( self.x_init, self.vx_init, self.ax_init ))
        initial_state_y = jnp.vstack(( self.y_init, self.vy_init, self.ay_init ))
        initial_state_z = jnp.vstack(( self.z_init, self.vz_init, self.az_init ))

        ########### Plan waypoints
        a =  self.a_obs_agent 
        b =  self.b_obs_agent 
        c =  self.c_obs_agent 

        a_sq =  self.a_obs_agent**2 
        b_sq =  self.b_obs_agent**2 
        c_sq =  self.c_obs_agent**2 

        Prob_waypoints = way_points.batch_crowd_nav( a, b, c, a_sq, b_sq, c_sq, self.a_obs, self.b_obs, self.c_obs, self.a_obs_agent, self.b_obs_agent, self.c_obs_agent, self.n_agents, self.num_obs, self.num_obs_dy, self.t_fin, self.num, self.maxiter_proj, self.weight_smoothness, self.rho_obs)
        key = random.PRNGKey(0)

        goal_state_x = jnp.vstack(( self.x_goal_fin, self.vx_goal, self.ax_goal ))
        goal_state_y = jnp.vstack(( self.y_goal_fin, self.vy_goal, self.ay_goal ))
        goal_state_z = jnp.vstack(( self.z_goal_fin, self.vz_goal, self.az_goal ))

        lamda_x = jnp.ones((self.n_agents, Prob_waypoints.nvar))
        lamda_y = jnp.ones((self.n_agents, Prob_waypoints.nvar))
        lamda_z = jnp.ones((self.n_agents, Prob_waypoints.nvar))

        x_obs_trajectory_dy, y_obs_trajectory_dy, z_obs_trajectory_dy = Prob_waypoints.compute_obs_dy_traj_prediction( self.x_agent, self.y_agent, self.z_agent) 

        # x_waypoint_init, y_waypoint_init, z_waypoint_init = Prob_waypoints.compute_optimization(key, initial_state_x, initial_state_y, initial_state_z, goal_state_x, goal_state_y, goal_state_z, self.x_goal, self.y_goal, self.z_goal, lamda_x, lamda_y, lamda_z, x_obs_trajectory_dy, y_obs_trajectory_dy, z_obs_trajectory_dy, self.x_agent, self.y_agent, self.z_agent) 
        
        x_waypoint = vmap(self.interp_signal)(self.x_waypoint)
        y_waypoint = vmap(self.interp_signal)(self.y_waypoint)
        z_waypoint = vmap(self.interp_signal)(self.z_waypoint)

        ############## MPC setting

        Prob = mpc_expert_local_sample.batch_crowd_nav( self.a_obs, self.b_obs, self.c_obs, self.a_obs_agent, self.b_obs_agent, self.c_obs_agent, self.n_agents, self.v_max, self.v_min, self.a_max, self.num_obs, self.num_obs_dy, self.num_obs_dynamic, self.t_fin, self.num, self.num_batch, self.maxiter_proj, self.maxiter_cem, self.weight_smoothness, self.waypoint_shape, self.v_des, self.rho_obs, self.rho_ineq, self.num_sample)
        
        lamda_x = jnp.zeros(( self.n_agents,  Prob.nvar))
        lamda_y = jnp.zeros(( self.n_agents,  Prob.nvar))
        lamda_z = jnp.zeros(( self.n_agents,  Prob.nvar))
        
        arc_length, arc_vec, x_diff, y_diff, z_diff = Prob.path_spline(x_waypoint, y_waypoint, z_waypoint)

        x_guess_warm, y_guess_warm, z_guess_warm = Prob.compute_traj_warm(initial_state_x, initial_state_y, initial_state_z,  x_waypoint, y_waypoint, z_waypoint, arc_vec, x_diff, y_diff, z_diff, self.v_des)

        plt.figure()

        plt.plot(x_waypoint[0,:], y_waypoint[0,:] , color = "orange")
        plt.plot(x_waypoint[1,:], y_waypoint[1,:], color = "green" )
        plt.plot(x_waypoint[2,:], y_waypoint[2,:], color = "brown")
        plt.plot(x_waypoint[3,:], y_waypoint[3,:], color = "yellow")
        plt.show()

        x_best = self.x_init[:, jnp.newaxis] * jnp.ones((self.n_agents, self.maxiter_mpc))
        y_best = self.y_init[:, jnp.newaxis] * jnp.ones((self.n_agents, self.maxiter_mpc))
        z_best = self.z_init[:, jnp.newaxis] * jnp.ones((self.n_agents, self.maxiter_mpc))

        counter = 1
        k_current = 1

        vx_obs = 0.0 * jnp.ones(self.num_obs_dynamic)
        vy_obs = 0.0 * jnp.ones(self.num_obs_dynamic)
        vz_obs = 0.0 * jnp.ones(self.num_obs_dynamic)

        save_iteration = jnp.zeros(self.n_agents)

        x_save = []
        y_save = []
        z_save = []

        x_init_save = []
        y_init_save = []
        z_init_save = []

        x_obs_st_save = []
        y_obs_st_save = []
        z_obs_st_save = []

        x_obs_dy_save = []
        y_obs_dy_save = []
        z_obs_dy_save = []

        x_save_copy = []
        y_save_copy = []
        z_save_copy = []

        x_init_save_copy = []
        y_init_save_copy = []
        z_init_save_copy = []

        x_obs_st_save_copy = []
        y_obs_st_save_copy = []
        z_obs_st_save_copy = []

        x_obs_dy_save_copy = []
        y_obs_dy_save_copy = []
        z_obs_dy_save_copy = []

        x_obs_dy_save_init_copy = []
        y_obs_dy_save_init_copy = []
        z_obs_dy_save_init_copy = []

        counter = 0
        idx_finish = self.maxiter_mpc* jnp.ones(self.n_agents)
        Total_time = 0

        Time_save = []


        for j in range(self.maxiter_mpc):

            initial_state_x = jnp.vstack(( self.x_init, self.vx_init, self.ax_init ))
            initial_state_y = jnp.vstack(( self.y_init, self.vy_init, self.ay_init ))
            initial_state_z = jnp.vstack(( self.z_init, self.vz_init, self.az_init ))

            prev_n = self.n_agents
        
            start = time.time()

            x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x_obs_dynamic_trajectory, y_obs_dynamic_trajectory, z_obs_dynamic_trajectory, x_obs_dynamic_trajectory_save, y_obs_dynamic_trajectory_save, z_obs_dynamic_trajectory_save  = Prob.compute_obs_traj_prediction( self.x_obs_dynamic, self.y_obs_dynamic, self.z_obs_dynamic, self.vx_obs_dynamic, self.vy_obs_dynamic, self.vz_obs_dynamic, jnp.asarray(self.x_init_obs).flatten(), jnp.asarray(self.y_init_obs).flatten(), jnp.asarray(self.z_init_obs).flatten(), self.vx_obs, self.vy_obs , self.vz_obs, self.x_init, self.y_init, self.z_init)
            
            sol_x_bar, sol_y_bar, sol_z_bar, x_guess, y_guess, z_guess, xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess, x_fin, y_fin, z_fin, x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent = Prob.compute_traj_guess(initial_state_x, initial_state_y, initial_state_z, x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x_waypoint, y_waypoint, z_waypoint, arc_vec, x_diff, y_diff, z_diff, self.v_des, self.x_goal, self.y_goal, self.z_goal, x_guess_warm , y_guess_warm,  z_guess_warm,  )
                                                                                        
            c_x_best, c_y_best, c_z_best, x, y, z, xdot, ydot, zdot, xddot, yddot, zddot, x_guess_warm, y_guess_warm , z_guess_warm = Prob.Compute_CEM( initial_state_x, initial_state_y, initial_state_z, x_fin, y_fin, z_fin, lamda_x, lamda_y, lamda_z, x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x_obs_dynamic_trajectory, y_obs_dynamic_trajectory, z_obs_dynamic_trajectory, x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent, sol_x_bar, sol_y_bar, sol_z_bar, x_guess, y_guess,  z_guess,     xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess, arc_vec, x_waypoint, y_waypoint, z_waypoint ) 
            
            vx_control, vy_control, vz_control, ax_control, ay_control, az_control = Prob.compute_controls(c_x_best, c_y_best, c_z_best)

            #############################################
            Dist = jnp.round( (self.x_goal_fin - self.x_init)**2 + (self.y_goal_fin - self.y_init)**2   + (self.z_goal_fin - self.z_init)**2, 2)
            finished_mask = Dist <= (0.8)**2
            finished_ids = self.active_ids[finished_mask]

            K_new = int(finished_ids.size)
        
            self.active_mask = self.active_mask.at[finished_mask].set(False)
            mask = self.active_mask[:, None]
            masked = ~self.active_mask

            Prob.set_active_mask(self.active_mask)
            
            if K_new > counter: 

                print("one agent is near final point", j ,"mpc_iter_ j", counter, "counter" )         

                # self.x_goal_fin = self.x_goal_fin.at[finished_mask].set(self.x_dummy[finished_mask, 1])
                # self.y_goal_fin = self.y_goal_fin.at[finished_mask].set(self.y_dummy[finished_mask, 1])
                # self.z_goal_fin = self.z_goal_fin.at[finished_mask].set(self.z_dummy[finished_mask, 1])

                # self.x_init = self.x_init.at[finished_mask].set(self.x_dummy[finished_mask,0])
                # self.y_init = self.y_init.at[finished_mask].set(self.y_dummy[finished_mask,0])
                # self.z_init = self.z_init.at[finished_mask].set(self.z_dummy[finished_mask,0]) 
    
                counter = jnp.sum(~self.active_mask)
                K_new = counter
                
                idx_finish = idx_finish.at[finished_mask].set(int(j))
                save_iteration = save_iteration.at[finished_mask].set(j)

                if counter == self.n_agents:
                    print("All agents reached to their goals!" )
                    break
            
                # x_guess_warm = x_guess_warm.at[finished_mask, :].set(self.x_dummy[finished_mask, :])
                # y_guess_warm = y_guess_warm.at[finished_mask, :].set(self.y_dummy[finished_mask, :])
                # z_guess_warm = z_guess_warm.at[finished_mask, :].set(self.z_dummy[finished_mask, :])

                # x_waypoint = x_waypoint.at[finished_mask, :].set(self.x_waypoint_dummy[finished_mask, :])
                # y_waypoint = y_waypoint.at[finished_mask, :].set(self.y_waypoint_dummy[finished_mask, :])
                # z_waypoint = z_waypoint.at[finished_mask, :].set(self.z_waypoint_dummy[finished_mask, :])

                # arc_length, arc_vec, x_diff, y_diff, z_diff = Prob.path_spline(x_waypoint, y_waypoint, z_waypoint)

                # self.x_obs_dynamic = self.x_obs_dynamic.at[finished_mask].set(x[finished_mask, 0])
                # self.y_obs_dynamic = self.y_obs_dynamic.at[finished_mask].set(y[finished_mask, 0])
                # self.z_obs_dynamic = self.z_obs_dynamic.at[finished_mask].set(z[finished_mask, 0])

                # x0 = x[finished_mask, 0]
                # y0 = y[finished_mask, 0]
                # z0 = z[finished_mask, 0]

                # xg = x_fin[finished_mask]
                # yg = y_fin[finished_mask]
                # zg = z_fin[finished_mask]

                # ux, uy, uz, d, steps_needed, moving = self._unit(x0, xg, y0, yg, z0, zg)

                # vx_obs = 0.0 * jnp.ones(self.num_obs_dynamic)
                # vy_obs = 0.0 * jnp.ones(self.num_obs_dynamic)
                # vz_obs = 0.0 * jnp.ones(self.num_obs_dynamic)

                # self.vx_obs_dynamic = self.vx_obs_dynamic.at[finished_mask,:].set(xdot[finished_mask, :])
                # self.vy_obs_dynamic = self.vy_obs_dynamic.at[finished_mask,:].set(ydot[finished_mask, :])
                # self.vz_obs_dynamic = self.vz_obs_dynamic.at[finished_mask,:].set(zdot[finished_mask, :])

            if counter == self.n_agents-1:
                index = jnp.where(self.active_mask)[0][0]
                vx_control = jnp.mean(xdot[index, 0:20], axis = 0) 
                vy_control = jnp.mean(ydot[index, 0:20], axis = 0) 
                vz_control = jnp.mean(zdot[index, 0:20], axis = 0) 

                ax_control = jnp.mean(xddot[index, 0:20], axis = 0) 
                ay_control = jnp.mean(yddot[index, 0:20], axis = 0) 
                az_control = jnp.mean(zddot[index, 0:20], axis = 0) 
            # print(Dist, "dist")
            
            m_control = self.active_mask.astype(vx_control.dtype)  # (N,)

            vx_control = m_control * vx_control
            vy_control = m_control * vy_control
            vz_control = m_control * vz_control
            ax_control = m_control * ax_control
            ay_control = m_control * ay_control
            az_control = m_control * az_control

            # print(vx_control, "vx_control")

            # self.x_init = self.x_init.at[finished_mask].set(self.x_dummy[finished_mask,0])
            # self.y_init = self.y_init.at[finished_mask].set(self.y_dummy[finished_mask,0])
            # self.z_init = self.z_init.at[finished_mask].set(self.z_dummy[finished_mask,0])

            self.x_init = self.x_init + self.t_update * vx_control
            self.y_init = self.y_init + self.t_update * vy_control
            self.z_init = self.z_init + self.t_update * vz_control


            self.vx_init = vx_control
            self.vy_init = vy_control
            self.vz_init = vz_control

            self.ax_init = ax_control
            self.ay_init = ay_control
            self.az_init = az_control

            


            Total_time = Total_time + time.time() - start
            Time_save.append(Total_time)

            
            print (time.time() - start, "Computation.time")

            x_save = np.array

            # self.x_obs_dynamic = self.x_obs_dynamic + vx_obs * self.t_update
            # self.y_obs_dynamic = self.y_obs_dynamic + vy_obs * self.t_update
            # self.z_obs_dynamic = self.z_obs_dynamic + vz_obs * self.t_update

            # vx_obs = self.vx_obs_dynamic[:,0]
            # vy_obs = self.vy_obs_dynamic[:,0]
            # vz_obs = self.vz_obs_dynamic[:,0]

            # self.vx_obs_dynamic = jnp.hstack(( self.vx_obs_dynamic[:,1:], self.vx_obs_dynamic[:, -1:]))
            # self.vy_obs_dynamic = jnp.hstack(( self.vy_obs_dynamic[:,1:], self.vy_obs_dynamic[:, -1:]))
            # self.vx_obs_dynamic = jnp.hstack(( self.vx_obs_dynamic[:,1:], jnp.zeros((self.num_obs_dy, 1)) ))
            # self.vy_obs_dynamic = jnp.hstack(( self.vy_obs_dynamic[:,1:], jnp.zeros((self.num_obs_dy, 1)) ))
            # self.vz_obs_dynamic = jnp.hstack(( self.vz_obs_dynamic[:,1:], jnp.zeros((self.num_obs_dy, 1)) ))

            
            x_best = x_best.at[:,j+1].set(self.x_init)
            y_best = y_best.at[:,j+1].set(self.y_init)
            z_best = z_best.at[:,j+1].set(self.z_init)


            x_obs_st_save_copy.append(self.x_init_obs)
            y_obs_st_save_copy.append(self.y_init_obs)
            z_obs_st_save_copy.append(self.z_init_obs)

            x_obs_dy_save_copy.append(x_obs_dynamic_trajectory_save)
            y_obs_dy_save_copy.append(y_obs_dynamic_trajectory_save)
            z_obs_dy_save_copy.append(z_obs_dynamic_trajectory_save)
            
            x_save_copy.append(x)
            y_save_copy.append(y)
            z_save_copy.append(z)

            x_init_save_copy.append(self.x_init)
            y_init_save_copy.append(self.y_init)
            z_init_save_copy.append(self.z_init)

            x_obs_dy_save_init_copy.append(self.x_obs_dynamic)
            y_obs_dy_save_init_copy.append(self.y_obs_dynamic)
            z_obs_dy_save_init_copy.append(self.z_obs_dynamic)

            
        print(Total_time, "Total time")
        x_obs_st_save_copy, y_obs_st_save_copy, z_obs_st_save_copy = np.array(x_obs_st_save_copy), np.array(y_obs_st_save_copy), np.array(z_obs_st_save_copy)
        x_obs_dy_save_copy, y_obs_dy_save_copy, z_obs_dy_save_copy = np.array(x_obs_dy_save_copy), np.array(y_obs_dy_save_copy), np.array(z_obs_dy_save_copy)
        x_save_copy, y_save_copy, z_save_copy = np.array(x_save_copy), np.array(y_save_copy), np.array(z_save_copy)
        x_init_save_copy, y_init_save_copy, z_init_save_copy = np.array(x_init_save_copy), np.array(y_init_save_copy), np.array(z_init_save_copy)
        x_obs_dy_save_init_copy, y_obs_dy_save_init_copy, z_obs_dy_save_init_copy = np.array(x_obs_dy_save_init_copy), np.array(y_obs_dy_save_init_copy), np.array(z_obs_dy_save_init_copy)

        Time_save = np.array(Time_save)

        idx_finish_type = idx_finish.astype(int)

        # x_save = np.concatenate([x_save_copy[:idx_finish_type[0]], x_obs_dy_save_copy[idx_finish_type[0]:]], axis=0).transpose(1, 2, 0)
        # y_save = np.concatenate([y_save_copy[:idx_finish_type[0]], y_obs_dy_save_copy[idx_finish_type[0]:]], axis=0).transpose(1, 2, 0)
        # z_save = np.concatenate([z_save_copy[:idx_finish_type[0]], z_obs_dy_save_copy[idx_finish_type[0]:]], axis=0).transpose(1, 2, 0)

        x_save = x_save_copy.transpose(1, 2, 0)#[:idx_finish_type[0]], x_obs_dy_save_copy[idx_finish_type[0]:]], axis=0).transpose(1, 2, 0)
        y_save = y_save_copy.transpose(1, 2, 0)#[:idx_finish_type[0]], y_obs_dy_save_copy[idx_finish_type[0]:]], axis=0).transpose(1, 2, 0)
        z_save = z_save_copy.transpose(1, 2, 0)#[:idx_finish_type[0]], z_obs_dy_save_copy[idx_finish_type[0]:]], axis=0).transpose(1, 2, 0)

        
        x_mpc_hist, y_mpc_hist, z_mpc_hist = x_save, y_save, z_save
        No = "4_robots_30obs_case_10"

        np.save("x_hist_save_" + str(No) + ".npy", x_init_save)
        np.save("y_hist_save_" + str(No) + ".npy", y_init_save)
        np.save("z_hist_save_" + str(No) + ".npy", z_init_save)

        np.save("x_waypoint_save_" + str(No) + ".npy", x_waypoint)
        np.save("y_waypoint_save_" + str(No) + ".npy", y_waypoint)
        np.save("z_waypoint_save_" + str(No) + ".npy", z_waypoint)

        np.save("x_obs_save_" + str(No) + ".npy", x_obs_st_save_copy)
        np.save("y_obs_save_" + str(No) + ".npy", y_obs_st_save_copy)
        np.save("z_obs_save_" + str(No) + ".npy", z_obs_st_save_copy)

        np.save("x_mpc_hist_" + str(No) + ".npy", np.array(x_mpc_hist))
        np.save("y_mpc_hist_" + str(No) + ".npy", np.array(y_mpc_hist))
        np.save("z_mpc_hist_" + str(No) + ".npy", np.array(z_mpc_hist))

        np.save("Time_save" + str(No) + ".npy", Time_save)

        print(np.shape(x_save), "x_mpc_hist")


        x_bestt = np.array(x_best)
        y_bestt = np.array(y_best)
        z_bestt = np.array(z_best)

        idx_finish = np.array(idx_finish)

        NO_agents_coll, NO_static_coll, NO_dynamic_coll =  compute_no_collisions( jnp.array(x_init_save_copy), jnp.array(y_init_save_copy), jnp.array(z_init_save_copy), x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, self.n_agents, self.num_obs, self.a_obs_agent, self.b_obs_agent, self.c_obs_agent, self.a_obs, self.b_obs, self.c_obs )
        

        # plt.plot(self.x_waypoint[0,:], self.y_waypoint[0,:] , color = "orange")
        # plt.plot(self.x_waypoint[1,:], self.y_waypoint[1,:], color = "green" )
        # plt.plot(self.x_waypoint[2,:], self.y_waypoint[2,:], color = "brown")
        # plt.plot(self.x_waypoint[3,:], self.y_waypoint[3,:], color = "yellow")

        # plt.plot(x[0,:], y[0,:] , color = "red")
        # plt.plot(x[1,:], y[1,:], color = "blue" )
        # plt.plot(x[2,:], y[2,:], color = "cyan")
        # plt.plot(x[3,:], y[3,:], color = "black")
            
        # plt.show()

        th = np.linspace(0, 2*np.pi, 100)


        plt.plot(x_bestt[0,:int(idx_finish[0])], y_bestt[0,:int(idx_finish[0])] , color = "red")
        plt.plot(x_bestt[1,:int(idx_finish[1])], y_bestt[1,:int(idx_finish[1])], color = "blue" )
        plt.plot(x_bestt[2,:int(idx_finish[2])], y_bestt[2,:int(idx_finish[2])], color = "pink")
        plt.plot(x_bestt[3,:int(idx_finish[3])], y_bestt[3,:int(idx_finish[3])], color = "navy")
        # plt.plot(x_bestt[4,:int(idx_finish[4])], y_bestt[4,:int(idx_finish[4])] , color = "green")
        # plt.plot(x_bestt[5,:int(idx_finish[5])], y_bestt[5,:int(idx_finish[5])], color = "lime" )
        # plt.plot(x_bestt[6,:int(idx_finish[6])], y_bestt[6,:int(idx_finish[6])], color = "coral")
        # plt.plot(x_bestt[7,:int(idx_finish[7])], y_bestt[7,:int(idx_finish[7])], color = "grey")
        # plt.plot(x_bestt[8,:int(idx_finish[8])], y_bestt[8,:int(idx_finish[8])] , color = "lightblue")
        # plt.plot(x_bestt[9,:int(idx_finish[9])], y_bestt[9,:int(idx_finish[9])], color = "brown" )
        # plt.plot(x_bestt[10,:int(idx_finish[10])], y_bestt[10,:int(idx_finish[10])], color = "hotpink")
        # plt.plot(x_bestt[11,:int(idx_finish[11])], y_bestt[11,:int(idx_finish[11])], color = "purple")
        # plt.plot(x_bestt[12,:int(idx_finish[12])], y_bestt[12,:int(idx_finish[12])] , color = "olive")
        # plt.plot(x_bestt[13,:int(idx_finish[13])], y_bestt[13,:int(idx_finish[13])], color = "orange" )
        # plt.plot(x_bestt[14,:int(idx_finish[14])], y_bestt[14,:int(idx_finish[14])], color = "cyan")
        # plt.plot(x_bestt[15,:int(idx_finish[15])], y_bestt[15,:int(idx_finish[15])], color = "black")

        # plt.plot(self.x_waypoint[0,:], self.y_waypoint[0,:] , color = "orange")
        # plt.plot(self.x_waypoint[1,:], self.y_waypoint[1,:], color = "green" )
        # plt.plot(self.x_waypoint[2,:], self.y_waypoint[2,:], color = "brown")
        # plt.plot(self.x_waypoint[3,:], self.y_waypoint[3,:], color = "yellow")

        # plt.plot(x_obs_trajectory[0,:].T + (self.a_obs - 0.04) * np.cos(th) , y_obs_trajectory[0,:].T + (self.b_obs - 0.04) * np.sin(th) )
        # plt.plot(x_obs_trajectory[1,:].T + (self.a_obs - 0.04) * np.cos(th) , y_obs_trajectory[1,:].T + (self.b_obs - 0.04) * np.sin(th) )
        # plt.plot(x_obs_trajectory[2,:].T + (self.a_obs - 0.04) * np.cos(th) , y_obs_trajectory[2,:].T + (self.b_obs - 0.04) * np.sin(th) )

        plt.show()


        

        



if __name__ == "__main__":

    motion_planning = planning_traj()

    obst_config_file = "/home/fatemeh/multi-agent/obst_configs_4.json"
    with open(obst_config_file, 'r') as f:
        all_configs = json.load(f)

    for config in all_configs["configs"]:
        motion_planning.data_prep(config)
        motion_planning.planner()


    



