import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random
import bernstein_coeff_order10_arbitinterval
import time
import jax
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt 
from scipy.interpolate import BarycentricInterpolator
from scipy.interpolate import UnivariateSpline
from jax import lax
import jax.lax as lax




class batch_crowd_nav():

    def __init__(self,  a_obs, b_obs, c_obs, a_obs_agent, b_obs_agent, c_obs_agent, n_agents, v_max, v_min, a_max, num_obs, num_obs_dy, num_obs_dynamic, t_fin, num, num_batch, maxiter_proj, maxiter_cem, weight_smoothness, way_point_shape, v_des, rho_obs, rho_ineq, num_sample):
        
        self.num_obs_dynamic = num_obs_dynamic

        self.a_obs = a_obs
        self.b_obs = b_obs
        self.c_obs = c_obs

        self.a_obs_agent = a_obs_agent
        self.b_obs_agent = b_obs_agent
        self.c_obs_agent = c_obs_agent

        self.num_obs = num_obs
        self.n_agents = n_agents

        self.v_max = v_max
        self.v_min = v_min
        self.a_max = a_max

        self.rho_obs = 1
        self.rho_ineq = 1
        self.rho_proj = 1
        self.rho_track = 1

        self.t_fin = t_fin
        self.num = num
        self.num_batch = num_batch
        self.maxiter_proj = maxiter_proj
        self.maxiter_cem = maxiter_cem
        self.weight_smoothness = weight_smoothness
        self.way_point_shape = way_point_shape
        self.v_des = v_des
        self.num_sample_warm = 1
        self.num_sample = num_sample
        self.initial_up_sampling = 1
        self.num_closest = 3
        self.num_obs_dy = num_obs_dy

        tot_time = np.linspace(0, self.t_fin, self.num)
        tot_time_copy = tot_time.reshape(self.num, 1)

        self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
        
        self.tot_time = tot_time
        
        self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)
        self.nvar = jnp.shape(self.P_jax)[1]

        self.A_projection = jnp.identity( self.nvar)
        self.A_eq = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0] ,  self.P_jax[-1] ))
        # self.A_eq = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0]  ))
        
        self.cost_x =  jnp.dot(self.P_jax.T, self.P_jax)  + 1*jnp.dot(self.Pddot_jax.T, self.Pddot_jax) + 0.00001 * jnp.identity(self.nvar)
        self.cost_y = self.cost_x
        self.cost_mat_x = self.cost_x
        self.cost_mat_inv_x = jnp.linalg.inv(self.cost_mat_x)
        self.cost_mat_y = self.cost_y
        self.cost_mat_inv_y = jnp.linalg.inv(self.cost_mat_y)
        self.cost_mat_inv_z = jnp.linalg.inv(self.cost_mat_y)
        self.setup_problem(self.num_obs, self.num_obs_dy, self.n_agents )

        self.compute_dynamic_obs = jit(jax.vmap(self.compute_dynamic_obs_per_agent, in_axes = (0,0,0, None, None, None)))

    ##############################################
    def setup_problem(self, num_obs, num_obs_dy, n_agents):
        self.num_obs = num_obs
        self.num_obs_dy = num_obs_dy
        self.n_agents = n_agents

        self.A_v = self.Pdot_jax 
        self.A_acc = self.Pddot_jax
        self.A_obs = jnp.tile(self.P_jax, (self.num_obs+ self.num_obs_dynamic + self.num_closest, 1))
        # self.A_obs = jnp.tile(self.P_jax, (self.num_obs, 1))

        self.cost_x_projection = 1.0*self.rho_proj * jnp.dot( self.A_projection.T, self.A_projection) + 1.0*self.rho_obs*jnp.dot(self.A_obs.T, self.A_obs)+1.0*self.rho_ineq*jnp.dot(self.A_acc.T, self.A_acc)+1.0*self.rho_ineq*jnp.dot(self.A_v.T, self.A_v)
        self.cost_mat_x_projection = jnp.vstack((  jnp.hstack(( self.cost_x_projection, self.A_eq.T )), jnp.hstack(( self.A_eq, jnp.zeros(( jnp.shape(self.A_eq)[0], jnp.shape(self.A_eq)[0] )) )) ))
        self.cost_mat_inv_x_projection = jnp.linalg.inv(self.cost_mat_x_projection)
        self.cost_mat_inv_y_projection = self.cost_mat_inv_x_projection
        self.cost_mat_inv_z_projection = self.cost_mat_inv_x_projection


        self.A_projection = jnp.identity( self.nvar)
        self.A_eq = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0] ,  self.P_jax[-1] ))
        # self.A_eq = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0]  ))

        # --- index matrix to gather "all rows except i" ---
        J = jnp.arange(self.num_closest)                    # (n_agents-1,)
        I = jnp.arange(self.n_agents).reshape(-1, 1)         # (n_agents, 1)
        # idx[i, j] = j if j < i else j+1
        self.other_idx = J + (J >= I).astype(J.dtype)        # (n_agents, n_agents-1)

        self.active_mask = jnp.ones((self.n_agents,), dtype=bool)
        self.x_fixed = jnp.zeros((self.n_agents, self.num), dtype=self.P_jax.dtype)
        self.y_fixed = jnp.zeros((self.n_agents, self.num), dtype=self.P_jax.dtype)
        self.z_fixed = jnp.zeros((self.n_agents, self.num), dtype=self.P_jax.dtype)

    ######## Do not jit
    def set_active_mask(self, mask):
        self.active_mask = mask

    ######################
    @partial(jit, static_argnums=(0,))
    def update_n_agents(self, n_agents):
        self.setup_problem( self.num_obs, self.num_obs_dy, self.n_agents)

    #################
    @partial(jit, static_argnums=(0,))
    def compute_b_part(self, sol_x_bar, sol_y_bar, sol_z_bar, d_obs, alpha_obs, beta_obs, x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x_obs_trajectory_dy, y_obs_trajectory_dy, z_obs_trajectory_dy, x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent):
  
        d_obs_st = d_obs[:, :self.num*(self.num_obs)]
        d_obs_dy = d_obs[:, self.num*(self.num_obs):(self.num*(self.num_obs) + self.num*self.num_obs_dynamic)]
        d_obs_agent = d_obs[:, (self.num*(self.num_obs) + self.num*self.num_obs_dynamic):]

        alpha_obs_st = alpha_obs[ :,  :self.num*(self.num_obs)]
        alpha_obs_dy = alpha_obs[:, self.num*(self.num_obs):(self.num*(self.num_obs) + self.num*self.num_obs_dynamic)]
        alpha_obs_agent = alpha_obs[ :, (self.num*(self.num_obs) + self.num*self.num_obs_dynamic):]
        
        beta_obs_st = beta_obs[ :,  :self.num*(self.num_obs)]
        beta_obs_dy = beta_obs[:, self.num*(self.num_obs):(self.num*(self.num_obs) + self.num*self.num_obs_dynamic)]
        beta_obs_agent = beta_obs[ :, (self.num*(self.num_obs) + self.num*self.num_obs_dynamic):]
      
        temp_x_obs_st = d_obs_st*jnp.cos(alpha_obs_st)  * jnp.sin(beta_obs_st) * self.a_obs
        b_obs_x_st =  x_obs_trajectory.reshape(self.num*self.num_obs )+temp_x_obs_st
            
        temp_y_obs_st = d_obs_st*jnp.sin(alpha_obs_st)* jnp.sin(beta_obs_st) * self.b_obs
        b_obs_y_st = y_obs_trajectory.reshape(self.num*self.num_obs ) +temp_y_obs_st

        temp_z_obs_st = d_obs_st* jnp.cos(beta_obs_st) * self.c_obs
        b_obs_z_st = z_obs_trajectory.reshape(self.num*self.num_obs ) +temp_z_obs_st

        temp_x_obs_dy = d_obs_dy*jnp.cos(alpha_obs_dy) * jnp.sin(beta_obs_dy)* self.a_obs_agent
        b_obs_x_dy =  x_obs_trajectory_dy.reshape(self.n_agents, self.num*self.num_obs_dynamic )+temp_x_obs_dy
            
        temp_y_obs_dy = d_obs_dy*jnp.sin(alpha_obs_dy)* jnp.sin(beta_obs_dy)* self.b_obs_agent
        b_obs_y_dy = y_obs_trajectory_dy.reshape(self.n_agents, self.num*self.num_obs_dynamic ) +temp_y_obs_dy
        
        temp_z_obs_dy = d_obs_dy* jnp.cos(beta_obs_dy)* self.c_obs_agent
        b_obs_z_dy = z_obs_trajectory_dy.reshape(self.n_agents, self.num*self.num_obs_dynamic ) +temp_z_obs_dy

        temp_x_obs_agent = d_obs_agent*jnp.cos(alpha_obs_agent)* jnp.sin(beta_obs_agent)* self.a_obs_agent
        b_obs_x_agent = (x_obs_traj_agent.reshape(self.n_agents, self.num*(self.num_closest))) + temp_x_obs_agent

        temp_y_obs_agent = d_obs_agent*jnp.sin(alpha_obs_agent) * jnp.sin(beta_obs_agent)* self.b_obs_agent
        b_obs_y_agent = (y_obs_traj_agent.reshape(self.n_agents, self.num*(self.num_closest))) + temp_y_obs_agent

        temp_z_obs_agent = d_obs_agent* jnp.cos(beta_obs_agent)* self.c_obs_agent
        b_obs_z_agent = (z_obs_traj_agent.reshape(self.n_agents, self.num*(self.num_closest))) + temp_z_obs_agent

        b_obs_x = jnp.hstack((b_obs_x_st, b_obs_x_dy, b_obs_x_agent))
        b_obs_y = jnp.hstack((b_obs_y_st, b_obs_y_dy, b_obs_y_agent))
        b_obs_z = jnp.hstack((b_obs_z_st, b_obs_z_dy, b_obs_z_agent))

        return b_obs_x, b_obs_y, b_obs_z

    ########################################################
    @partial(jit, static_argnums=(0,))
    def compute_guess(self, x_guess, y_guess, z_guess):

        lincost_x = jnp.dot(self.P_jax.T, -x_guess.T).T
        lincost_y = jnp.dot(self.P_jax.T, -y_guess.T).T
        lincost_z = jnp.dot(self.P_jax.T, -z_guess.T).T

        sol_x = jnp.dot(self.cost_mat_inv_x, -lincost_x.T).T
        sol_y = jnp.dot(self.cost_mat_inv_y, -lincost_y.T).T
        sol_z = jnp.dot(self.cost_mat_inv_z, -lincost_z.T).T

        sol_x_bar = sol_x[:,0:self.nvar]
        sol_y_bar = sol_y[:,0:self.nvar]
        sol_z_bar = sol_z[:,0:self.nvar]
       
        x_guess = jnp.dot( self.P_jax, sol_x_bar.T).T
        xdot_guess = jnp.dot(self.Pdot_jax, sol_x_bar.T).T
        xddot_guess = jnp.dot(self.Pddot_jax, sol_x_bar.T).T

        y_guess = jnp.dot(self.P_jax, sol_y_bar.T).T
        ydot_guess = jnp.dot(self.Pdot_jax, sol_y_bar.T).T
        yddot_guess = jnp.dot( self.Pddot_jax, sol_y_bar.T).T

        z_guess = jnp.dot(self.P_jax, sol_z_bar.T).T
        zdot_guess = jnp.dot(self.Pdot_jax, sol_z_bar.T).T
        zddot_guess = jnp.dot( self.Pddot_jax, sol_z_bar.T).T

        return sol_x_bar, sol_y_bar, sol_z_bar, x_guess, y_guess, z_guess, xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess

    #################################################
    @partial(jit, static_argnums=(0))
    def compute_alpha_st(self, x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x_guess, y_guess, z_guess):

        wc_alpha_obs_temp_st = x_guess - x_obs_trajectory[:, jnp.newaxis]
        ws_alpha_obs_temp_st = y_guess - y_obs_trajectory[:, jnp.newaxis]

        wc_alpha_obs_temp_st = wc_alpha_obs_temp_st.transpose(1,0,2)
        ws_alpha_obs_temp_st = ws_alpha_obs_temp_st.transpose(1,0,2)

        wc_alpha_obs_st = wc_alpha_obs_temp_st.reshape(self.n_agents, self.num*(self.num_obs))
        ws_alpha_obs_st = ws_alpha_obs_temp_st.reshape(self.n_agents, self.num*(self.num_obs))

        alpha_obs_st = jnp.arctan2( ws_alpha_obs_st * self.a_obs, wc_alpha_obs_st * self.b_obs)

        wc_beta_obs_temp_st = (z_guess - z_obs_trajectory[:, jnp.newaxis])
        wc_beta_obs_temp_st = wc_beta_obs_temp_st.transpose(1,0,2)
        wc_beta_obs_st = wc_beta_obs_temp_st.reshape(self.n_agents, self.num*( self.num_obs))

        ws_beta_obs_st = wc_alpha_obs_st/jnp.cos(alpha_obs_st)
        beta_obs_st = jnp.arctan2( ws_beta_obs_st * self.c_obs, wc_beta_obs_st * self.a_obs)
        
        c1_d = 1.0*self.rho_obs*(self.a_obs**2*jnp.sin(beta_obs_st)**2 + self.c_obs**2*jnp.cos(beta_obs_st)**2 )
        c2_d = 1.0*self.rho_obs*(self.a_obs*wc_alpha_obs_st*jnp.cos(alpha_obs_st)*jnp.sin(beta_obs_st) + self.b_obs*ws_alpha_obs_st*jnp.sin(alpha_obs_st)*jnp.sin(beta_obs_st) + self.c_obs * wc_beta_obs_st*jnp.cos(beta_obs_st) )

        d_obs = c2_d/c1_d
        d_obs = jnp.maximum(jnp.ones((self.n_agents,  self.num*(self.num_obs)   )), d_obs   )

        res_x_obs_vec_st = wc_alpha_obs_st- self.a_obs *d_obs*jnp.cos(alpha_obs_st) *jnp.sin(beta_obs_st)
        res_y_obs_vec_st = ws_alpha_obs_st- self.b_obs *d_obs*jnp.sin(alpha_obs_st) *jnp.sin(beta_obs_st)
        res_z_obs_vec_st = wc_beta_obs_st - self.c_obs *d_obs * jnp.cos(beta_obs_st)


        return alpha_obs_st, beta_obs_st, d_obs, wc_alpha_obs_st, ws_alpha_obs_st, wc_beta_obs_st, ws_beta_obs_st, res_x_obs_vec_st, res_y_obs_vec_st, res_z_obs_vec_st
               
    ##############################################
    @partial(jit, static_argnums=(0,))
    def compute_alpha_dy(self, x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x_guess, y_guess, z_guess):

        wc_alpha_obs_temp = x_guess[:, jnp.newaxis, :] - x_obs_trajectory
        ws_alpha_obs_temp = y_guess[:, jnp.newaxis, :] - y_obs_trajectory

        wc_alpha_obs_temp = wc_alpha_obs_temp
        ws_alpha_obs_temp = ws_alpha_obs_temp

        wc_alpha_obs = wc_alpha_obs_temp.reshape(self.n_agents, self.num*(self.num_obs_dynamic))
        ws_alpha_obs = ws_alpha_obs_temp.reshape(self.n_agents, self.num*(self.num_obs_dynamic))

        alpha_obs = jnp.arctan2( ws_alpha_obs * self.a_obs_agent, wc_alpha_obs * self.a_obs_agent)

        wc_beta_obs_temp = (z_guess[:, jnp.newaxis, :] - z_obs_trajectory).transpose(1,0,2)
        wc_beta_obs = wc_beta_obs_temp.reshape(self.n_agents, self.num*( self.num_obs_dynamic))

        ws_beta_obs = wc_alpha_obs/jnp.cos(alpha_obs)
        beta_obs = jnp.arctan2( ws_beta_obs, wc_beta_obs)
        
        c1_d = 1.0*self.rho_obs*(self.a_obs_agent**2*jnp.sin(beta_obs)**2 + self.c_obs_agent**2*jnp.cos(beta_obs)**2 )
        c2_d = 1.0*self.rho_obs*(self.a_obs_agent*wc_alpha_obs*jnp.cos(alpha_obs)*jnp.sin(beta_obs) + self.b_obs_agent*ws_alpha_obs*jnp.sin(alpha_obs)*jnp.sin(beta_obs) + self.c_obs_agent*wc_beta_obs*jnp.cos(beta_obs)  )

        d_obs = c2_d/c1_d
        d_obs = jnp.maximum(jnp.ones((self.n_agents, self.num*(self.num_obs_dynamic )   )), d_obs   )

        res_x_obs_vec_agent = wc_alpha_obs-  self.a_obs_agent * d_obs*jnp.cos(alpha_obs) *jnp.sin(beta_obs)
        res_y_obs_vec_agent = ws_alpha_obs-  self.b_obs_agent * d_obs*jnp.sin(alpha_obs) *jnp.sin(beta_obs)
        res_z_obs_vec_agent = wc_beta_obs-  self.c_obs_agent * d_obs*jnp.cos(beta_obs)

        return alpha_obs, beta_obs, d_obs, wc_alpha_obs, ws_alpha_obs, wc_beta_obs, ws_beta_obs, res_x_obs_vec_agent, res_y_obs_vec_agent, res_z_obs_vec_agent

    #################################################
    @partial(jit, static_argnums=(0,))
    def compute_alpha_agent(self, x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x_guess, y_guess, z_guess):

        wc_alpha_obs_temp = x_guess[:, jnp.newaxis, :] - x_obs_trajectory
        ws_alpha_obs_temp = y_guess[:, jnp.newaxis, :] - y_obs_trajectory

        wc_alpha_obs_temp = wc_alpha_obs_temp
        ws_alpha_obs_temp = ws_alpha_obs_temp

        wc_alpha_obs = wc_alpha_obs_temp.reshape(self.n_agents, self.num*(self.num_closest))
        ws_alpha_obs = ws_alpha_obs_temp.reshape(self.n_agents, self.num*(self.num_closest))

        alpha_obs = jnp.arctan2( ws_alpha_obs * self.a_obs_agent, wc_alpha_obs * self.a_obs_agent)

        wc_beta_obs_temp = (z_guess[:, jnp.newaxis, :] - z_obs_trajectory).transpose(1,0,2)
        wc_beta_obs = wc_beta_obs_temp.reshape(self.n_agents, self.num*( self.num_closest))

        ws_beta_obs = wc_alpha_obs/jnp.cos(alpha_obs)
        beta_obs = jnp.arctan2( ws_beta_obs * self.c_obs_agent, wc_beta_obs * self.a_obs_agent)
        
        c1_d = 1.0*self.rho_obs*(self.a_obs_agent**2*jnp.sin(beta_obs)**2 + self.c_obs_agent**2*jnp.cos(beta_obs)**2 )
        c2_d = 1.0*self.rho_obs*(self.a_obs_agent*wc_alpha_obs*jnp.cos(alpha_obs)*jnp.sin(beta_obs) + self.b_obs_agent*ws_alpha_obs*jnp.sin(alpha_obs)*jnp.sin(beta_obs) + self.c_obs_agent*wc_beta_obs*jnp.cos(beta_obs)  )

        d_obs = c2_d/c1_d
        d_obs = jnp.maximum(jnp.ones((self.n_agents, self.num*(self.num_closest )   )), d_obs   )

        res_x_obs_vec_agent = wc_alpha_obs-  self.a_obs_agent * d_obs*jnp.cos(alpha_obs) *jnp.sin(beta_obs)
        res_y_obs_vec_agent = ws_alpha_obs-  self.b_obs_agent * d_obs*jnp.sin(alpha_obs) *jnp.sin(beta_obs)
        res_z_obs_vec_agent = wc_beta_obs-  self.c_obs_agent * d_obs*jnp.cos(beta_obs)

        return alpha_obs, beta_obs, d_obs, wc_alpha_obs, ws_alpha_obs, wc_beta_obs, ws_beta_obs, res_x_obs_vec_agent, res_y_obs_vec_agent, res_z_obs_vec_agent

    ############################
    @partial(jit, static_argnums=(0,))
    def compute_x_batch(self, b_eq_x, b_eq_y, b_eq_z, b_projection_x, b_projection_y, b_projection_z, b_obs_x, b_obs_y, b_obs_z, b_ax_ineq, b_ay_ineq, b_az_ineq, b_vx_ineq, b_vy_ineq, b_vz_ineq, lamda_x, lamda_y, lamda_z ):

        lincost_x = - 1.0* self.rho_proj * jnp.dot(self.A_projection.T, b_projection_x.T).T - lamda_x - 1.0*self.rho_ineq * jnp.dot(self.A_v.T, b_vx_ineq.T).T - 1.0*self.rho_ineq * jnp.dot( self.A_acc.T, b_ax_ineq.T).T  -1.0*self.rho_obs*jnp.dot( self.A_obs.T, b_obs_x.T).T 
        lincost_y = - 1.0* self.rho_proj * jnp.dot(self.A_projection.T, b_projection_y.T).T - lamda_y - 1.0*self.rho_ineq * jnp.dot(self.A_v.T, b_vy_ineq.T).T - 1.0*self.rho_ineq * jnp.dot( self.A_acc.T, b_ay_ineq.T).T  -1.0*self.rho_obs*jnp.dot( self.A_obs.T, b_obs_y.T).T #- 1.0*self.rho_track * jnp.einsum('ij,bjk->bik', self.A_track.T, b_try_ineq.transpose(0, 2, 1)).transpose(0, 2, 1)#- 1.0*self.rho_ineq * jnp.einsum('ij,bjk->bik', self.A_f.T, b_fy_ineq.transpose(0, 2, 1)).transpose(0, 2, 1)
        lincost_z = - 1.0* self.rho_proj * jnp.dot(self.A_projection.T, b_projection_z.T).T - lamda_z - 1.0*self.rho_ineq * jnp.dot(self.A_v.T, b_vz_ineq.T).T - 1.0*self.rho_ineq * jnp.dot( self.A_acc.T, b_az_ineq.T).T  -1.0*self.rho_obs*jnp.dot( self.A_obs.T, b_obs_z.T).T #- 1.0*self.rho_track * jnp.einsum('ij,bjk->bik', self.A_track.T, b_try_ineq.transpose(0, 2, 1)).transpose(0, 2, 1)#- 1.0*self.rho_ineq * jnp.einsum('ij,bjk->bik', self.A_f.T, b_fy_ineq.transpose(0, 2, 1)).transpose(0, 2, 1)

        sol_x = jnp.dot( self.cost_mat_inv_x_projection, jnp.hstack(( -lincost_x, b_eq_x )).T).T
        sol_y = jnp.dot( self.cost_mat_inv_y_projection, jnp.hstack(( -lincost_y, b_eq_y )).T).T
        sol_z = jnp.dot( self.cost_mat_inv_z_projection, jnp.hstack(( -lincost_z, b_eq_z )).T).T

        primal_sol_x = sol_x[:,0:self.nvar]
        primal_sol_y = sol_y[:,0:self.nvar]
        primal_sol_z = sol_z[:,0:self.nvar]

        x = jnp.dot(self.P_jax, primal_sol_x.T).T
        xdot = jnp.dot(self.Pdot_jax, primal_sol_x.T).T
        xddot = jnp.dot(self.Pddot_jax, primal_sol_x.T).T

        y = jnp.dot(self.P_jax, primal_sol_y.T).T
        ydot = jnp.dot(self.Pdot_jax, primal_sol_y.T).T
        yddot = jnp.dot(self.Pddot_jax, primal_sol_y.T).T

        z = jnp.dot(self.P_jax, primal_sol_z.T).T
        zdot = jnp.dot(self.Pdot_jax, primal_sol_z.T).T
        zddot = jnp.dot(self.Pddot_jax, primal_sol_z.T).T

        return primal_sol_x, primal_sol_y, primal_sol_z, x, xdot, xddot, y, ydot, yddot, z, zdot, zddot

    ##############################################
    @partial(jit, static_argnums=(0,))
    def compute_lamda(self, res_vx_vec, res_vy_vec, res_vz_vec, res_ax_vec, res_ay_vec, res_az_vec, res_x_obs_vec, res_y_obs_vec, res_z_obs_vec, lamda_x, lamda_y, lamda_z, res_vel_vec, res_acc_vec, res_obs_vec ):

        lamda_x = lamda_x  -1.0* self.rho_ineq* jnp.dot( self.A_v.T, res_vx_vec.T).T -1.0* self.rho_ineq* jnp.dot( self.A_acc.T, res_ax_vec.T).T  - 1.0*self.rho_obs* jnp.dot(self.A_obs.T, res_x_obs_vec.T).T #-1.0* self.rho_track* jnp.dot(self.A_track.T, res_trx_vec.T).T #-1.0* self.rho_ineq* jnp.dot( self.A_f.T, res_fx_vec.T).T
        lamda_y = lamda_y  -1.0* self.rho_ineq* jnp.dot( self.A_v.T, res_vy_vec.T).T -1.0* self.rho_ineq* jnp.dot( self.A_acc.T, res_ay_vec.T).T  - 1.0*self.rho_obs* jnp.dot(self.A_obs.T, res_y_obs_vec.T).T #-1.0* self.rho_track* jnp.dot(self.A_track.T, res_try_vec.T).T #-1.0* self.rho_ineq* jnp.dot( self.A_f.T, res_fy_vec.T).T
        lamda_z = lamda_z  -1.0* self.rho_ineq* jnp.dot( self.A_v.T, res_vz_vec.T).T -1.0* self.rho_ineq* jnp.dot( self.A_acc.T, res_az_vec.T).T  - 1.0*self.rho_obs* jnp.dot(self.A_obs.T, res_z_obs_vec.T).T #-1.0* self.rho_track* jnp.dot(self.A_track.T, res_try_vec.T).T #-1.0* self.rho_ineq* jnp.dot( self.A_f.T, res_fy_vec.T).T

        res_obs_vec = jnp.hstack(( res_x_obs_vec, res_y_obs_vec, res_z_obs_vec  ))

        res_norm = 1.0*jnp.linalg.norm(res_obs_vec, axis =1)+1.0*jnp.linalg.norm(res_acc_vec, axis =1)+1.0*jnp.linalg.norm(res_vel_vec, axis =1)

        return lamda_x, lamda_y, lamda_z, res_norm

    #######################################
    @partial(jit, static_argnums=(0,))
    def compute_obs_dy_traj_prediction( self, x_agent, y_agent, z_agent):

        m = self.active_mask[:, None]  # (N,1) for broadcast over T
        x_blend = jnp.where(m, x_agent, self.x_fixed)
        y_blend = jnp.where(m, y_agent, self.y_fixed)
        z_blend = jnp.where(m, z_agent, self.z_fixed)
        
        x_obs = self.build_obs(x_agent)
        y_obs = self.build_obs(y_agent)
        z_obs = self.build_obs(z_agent)

        x_start = x_agent[:, 0]
        y_start = y_agent[:, 0]
        z_start = z_agent[:, 0]

        dist_x = x_start[:, jnp.newaxis] - x_obs[:, :, 0]
        dist_y = y_start[:, jnp.newaxis] - y_obs[:, :, 0]
        dist_z = z_start[:, jnp.newaxis] - z_obs[:, :, 0]


        dist = jnp.sqrt(dist_x**2 + dist_y**2 + dist_z**2)

        K = min(self.num_closest, dist.shape[1])
        idx_closest = jnp.argsort(dist, axis=1)[:, :K]  # (n_agents, K)

        x_obs_dy_trajectory = jnp.take_along_axis(x_obs, idx_closest[:, :, None], axis=1)
        y_obs_dy_trajectory = jnp.take_along_axis(y_obs, idx_closest[:, :, None], axis=1)
        z_obs_dy_trajectory = jnp.take_along_axis(z_obs, idx_closest[:, :, None], axis=1)

        return x_obs_dy_trajectory, y_obs_dy_trajectory, z_obs_dy_trajectory

    ##############
    @partial(jit, static_argnums=(0,))
    def build_obs(self, data):

        return jnp.take(data, self.other_idx, axis=0)

    ################

    @partial(jit, static_argnums=(0,))
    def compute_dynamic_obs_per_agent(self, x_obs_init_dy, y_obs_init_dy, z_obs_init_dy, vx_obs_dy, vy_obs_dy, vz_obs_dy):

        x_temp_dy = x_obs_init_dy[:,jnp.newaxis] + vx_obs_dy * self.tot_time[:,jnp.newaxis].T
        x_obs_dynamic = x_temp_dy 

        y_temp_dy = y_obs_init_dy[:,jnp.newaxis] + vy_obs_dy *self.tot_time[:,jnp.newaxis].T
        y_obs_dynamic = y_temp_dy

        z_temp_dy = z_obs_init_dy[:,jnp.newaxis] + vz_obs_dy *self.tot_time[:,jnp.newaxis].T
        z_obs_dynamic = z_temp_dy

        return x_obs_dynamic, y_obs_dynamic, z_obs_dynamic
    ######################################
    @partial(jit, static_argnums=(0,))
    def compute_obs_traj_prediction(self, x_obs_init_dy, y_obs_init_dy, z_obs_init_dy, vx_obs_dy, vy_obs_dy, vz_obs_dy, x_obs_init, y_obs_init, z_obs_init, vx_obs, vy_obs, vz_obs, x_init, y_init, z_init):

        x_temp_1 = x_obs_init[:,jnp.newaxis]+vx_obs*self.tot_time[:,jnp.newaxis].T
        x_obs_trajectory = x_temp_1

        y_temp_1 = y_obs_init[:,jnp.newaxis]+vy_obs*self.tot_time[:,jnp.newaxis].T
        y_obs_trajectory = y_temp_1

        z_temp_1 = z_obs_init[:,jnp.newaxis]+vz_obs*self.tot_time[:,jnp.newaxis].T
        z_obs_trajectory = z_temp_1

        x_temp_dy = x_obs_init_dy[:,jnp.newaxis]+vx_obs_dy*self.tot_time[:,jnp.newaxis].T
        x_obs_trajectory_dy = x_temp_dy

        y_temp_dy = y_obs_init_dy[:,jnp.newaxis]+vy_obs_dy*self.tot_time[:,jnp.newaxis].T
        y_obs_trajectory_dy = y_temp_dy

        z_temp_dy = z_obs_init_dy[:,jnp.newaxis]+vz_obs_dy*self.tot_time[:,jnp.newaxis].T
        z_obs_trajectory_dy = z_temp_dy

        dist_x = x_init[:, jnp.newaxis] - x_obs_init_dy
        dist_y = y_init[:, jnp.newaxis] - y_obs_init_dy
        dist_z = z_init[:, jnp.newaxis] - z_obs_init_dy

        dist = jnp.sqrt(dist_x**2 + dist_y**2 + dist_z**2)

        K = min(self.num_obs_dy, dist.shape[1])
        idx_closest = jnp.argsort(dist, axis=1)[:, :K]  # (n_agents, K)

        x_obs_dy_close = jnp.take_along_axis(x_obs_init_dy[jnp.newaxis, :], idx_closest, axis=1)
        y_obs_dy_close = jnp.take_along_axis(y_obs_init_dy[jnp.newaxis, :], idx_closest, axis=1)
        z_obs_dy_close = jnp.take_along_axis(z_obs_init_dy[jnp.newaxis, :], idx_closest, axis=1)

        # print(jnp.shape(x_obs_dy_close),"sh")

        x_obs_dynamic, y_obs_dynamic, z_obs_dynamic = self.compute_dynamic_obs(x_obs_dy_close, y_obs_dy_close, z_obs_dy_close, vx_obs_dy, vy_obs_dy, vz_obs_dy)

        return x_obs_trajectory, y_obs_trajectory,z_obs_trajectory, x_obs_dynamic, y_obs_dynamic, z_obs_dynamic, x_obs_trajectory_dy, y_obs_trajectory_dy, z_obs_trajectory_dy
    
    ##################################
    @partial(jit, static_argnums=(0,),backend="gpu")
    def path_spline(self, x_waypoint, y_waypoint, z_waypoint):

        n_agents = x_waypoint.shape[0]

        x_diff = jnp.diff(x_waypoint, axis = 1)
        y_diff = jnp.diff(y_waypoint, axis = 1)
        z_diff = jnp.diff(z_waypoint, axis = 1)

        arc = jnp.cumsum(jnp.sqrt( x_diff**2 + y_diff**2 + z_diff**2), axis = 1)
        arc_length = arc[:, -1]

        arc_vec = jnp.linspace(jnp.zeros(n_agents), arc_length, self.way_point_shape).T

        return arc_length, arc_vec, x_diff, y_diff, z_diff

    ##################
    @partial(jit, static_argnums=(0,))
    def compute_traj_warm(self, initial_state_x, initial_state_y, initial_state_z, x_waypoint, y_waypoint, z_waypoint, arc_vec, x_diff, y_diff, z_diff, v_des):
        
        x_init = initial_state_x[0,:].flatten()
        y_init = initial_state_y[0,:].flatten()
        z_init = initial_state_z[0,:].flatten()

        dist = jnp.sqrt( (x_waypoint - x_init[:,jnp.newaxis])**2 + (y_waypoint - y_init[:,jnp.newaxis])**2 + (z_waypoint - z_init[:,jnp.newaxis])**2 )
        index = jnp.argmin(dist, axis=1).astype(jnp.int32)# jnp.argmin(dist, axis=1)
        W = arc_vec.shape[1]
        index = jnp.clip(index, 0, W-1)
        arc_point =   jnp.take_along_axis(arc_vec, index[:, None], axis=1).squeeze(1)

        look_ahead_point_path = arc_point+v_des*self.t_fin
        look_ahead_point_path = jnp.clip(look_ahead_point_path, arc_vec[:, 0], arc_vec[:, -1])
        index_final_path = jnp.argmin(jnp.abs(look_ahead_point_path[:,jnp.newaxis]-arc_vec) , axis = 1 )

        x_fin_path = jnp.take_along_axis(x_waypoint, index_final_path[:, None], axis=1).squeeze(1)  # (N,)
        y_fin_path = jnp.take_along_axis(y_waypoint, index_final_path[:, None], axis=1).squeeze(1)  # (N,)
        z_fin_path = jnp.take_along_axis(z_waypoint, index_final_path[:, None], axis=1).squeeze(1)
        #############################################################
        
        x_init_vec = x_init
        y_init_vec = y_init
        z_init_vec = z_init

        vx_init_vec = initial_state_x[1, :]
        vy_init_vec = initial_state_y[1, :]
        vz_init_vec = initial_state_z[1, :]

        ax_init_vec = initial_state_x[2, :]
        ay_init_vec = initial_state_y[2, :]
        az_init_vec = initial_state_z[2, :]


        x_fin_vec = x_fin_path
        y_fin_vec = y_fin_path
        z_fin_vec = z_fin_path

        b_eq_x = jnp.vstack((x_init_vec, vx_init_vec, ax_init_vec, x_fin_vec )).T
        b_eq_y = jnp.vstack((y_init_vec, vy_init_vec, ay_init_vec, y_fin_vec )).T
        b_eq_z = jnp.vstack((z_init_vec, vz_init_vec, az_init_vec, z_fin_vec )).T


        A_eq = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0], self.P_jax[-1]  ))
        cost_mat = jnp.vstack((  jnp.hstack(( jnp.dot(self.Pddot_jax.T, self.Pddot_jax), A_eq.T )), jnp.hstack(( A_eq, jnp.zeros(( jnp.shape(A_eq)[0], jnp.shape(A_eq)[0] )) )) ))
        cost_mat_inv = jnp.linalg.inv(cost_mat)

        N = x_init.shape[0]
        zerosN = jnp.zeros((N, self.nvar), dtype=self.P_jax.dtype)

        lincost_x = jnp.hstack(( -zerosN, b_eq_x ))
        lincost_y = jnp.hstack(( -zerosN, b_eq_y ))
        lincost_z = jnp.hstack(( -zerosN, b_eq_z ))
        
        sol_x = jnp.dot(cost_mat_inv, lincost_x.T).T
        sol_y = jnp.dot(cost_mat_inv, lincost_y.T).T
        sol_z = jnp.dot(cost_mat_inv, lincost_z.T).T

        sol_x_bar = sol_x[:,0:self.nvar]
        sol_y_bar = sol_y[:,0:self.nvar]
        sol_z_bar = sol_z[:,0:self.nvar]

        x_guess_warm = jnp.dot(self.P_jax, sol_x_bar.T).T
        y_guess_warm = jnp.dot(self.P_jax, sol_y_bar.T).T
        z_guess_warm = jnp.dot(self.P_jax, sol_z_bar.T).T

        return x_guess_warm, y_guess_warm, z_guess_warm

    ###########################################3

    @partial(jit, static_argnums=(0,))
    def compute_traj_guess(self, initial_state_x, initial_state_y, initial_state_z, x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x_waypoint, y_waypoint, z_waypoint, arc_vec, x_diff, y_diff, z_diff, v_des, x_f, y_f, z_f, x_guess_warm , y_guess_warm, z_guess_warm ):
        
        x_init = initial_state_x[0,:].flatten()
        y_init = initial_state_y[0,:].flatten()
        z_init = initial_state_z[0,:].flatten()

        dist = jnp.sqrt( (x_waypoint - x_init[:,jnp.newaxis])**2 + (y_waypoint - y_init[:,jnp.newaxis])**2 +(z_waypoint - z_init[:,jnp.newaxis])**2)
        index = jnp.argmin(dist, axis=1).astype(jnp.int32)# jnp.argmin(dist, axis=1)
        W = arc_vec.shape[1]
        index = jnp.clip(index, 0, W-1)
        arc_point =   jnp.take_along_axis(arc_vec, index[:, None], axis=1).squeeze(1)

        look_ahead_point_path = arc_point+v_des*self.t_fin
        look_ahead_point_path = jnp.clip(look_ahead_point_path, arc_vec[:, 0], arc_vec[:, -1])
        index_final_path = jnp.argmin(jnp.abs(look_ahead_point_path[:,jnp.newaxis]-arc_vec) , axis = 1 )

        x_fin_path = jnp.take_along_axis(x_waypoint, index_final_path[:, None], axis=1).squeeze(1)  # (N,)
        y_fin_path = jnp.take_along_axis(y_waypoint, index_final_path[:, None], axis=1).squeeze(1)  # (N,)
        z_fin_path = jnp.take_along_axis(z_waypoint, index_final_path[:, None], axis=1).squeeze(1)  # (N,)

        ##############################
        
        x_init_vec = x_init
        y_init_vec = y_init
        z_init_vec = z_init

        vx_init_vec = initial_state_x[1, :]
        vy_init_vec = initial_state_y[1, :]
        vz_init_vec = initial_state_z[1, :]

        ax_init_vec = initial_state_x[2, :]
        ay_init_vec = initial_state_y[2, :]
        az_init_vec = initial_state_z[2, :]

        x_fin_vec = x_fin_path
        y_fin_vec = y_fin_path
        z_fin_vec = z_fin_path

        ##########################

        b_eq_x = jnp.vstack((x_init_vec, vx_init_vec, ax_init_vec, x_fin_vec )).T
        b_eq_y = jnp.vstack((y_init_vec, vy_init_vec, ay_init_vec, y_fin_vec )).T
        b_eq_z = jnp.vstack((z_init_vec, vz_init_vec, az_init_vec, z_fin_vec )).T

        A_eq = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0],   self.P_jax[-1]  ))
        cost_mat = jnp.vstack((  jnp.hstack(( jnp.dot(self.Pddot_jax.T, self.Pddot_jax), A_eq.T )), jnp.hstack(( A_eq, jnp.zeros(( jnp.shape(A_eq)[0], jnp.shape(A_eq)[0] )) )) ))
        cost_mat_inv = jnp.linalg.inv(cost_mat)

        N = x_init.shape[0]
        zerosN = jnp.zeros((N, self.nvar), dtype=self.P_jax.dtype)

        lincost_x = jnp.hstack(( -zerosN, b_eq_x ))
        lincost_y = jnp.hstack(( -zerosN, b_eq_y ))
        lincost_z = jnp.hstack(( -zerosN, b_eq_z ))
        
        sol_x = jnp.dot(cost_mat_inv, lincost_x.T).T
        sol_y = jnp.dot(cost_mat_inv, lincost_y.T).T
        sol_z = jnp.dot(cost_mat_inv, lincost_z.T).T

        sol_x_bar = sol_x[:,0:self.nvar]
        sol_y_bar = sol_y[:,0:self.nvar]
        sol_z_bar = sol_z[:, :self.nvar]

        x_guess_temp = jnp.dot( self.P_jax, sol_x_bar.T).T
        y_guess_temp = jnp.dot( self.P_jax, sol_y_bar.T).T
        z_guess_temp = jnp.dot(self.P_jax, sol_z_bar.T).T

        x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent = self.compute_obs_dy_traj_prediction(x_guess_temp, y_guess_temp, z_guess_temp)       
        x_obs_traj_agent_warm, y_obs_traj_agent_warm, z_obs_traj_agent_warm = self.compute_obs_dy_traj_prediction(x_guess_warm, y_guess_warm, z_guess_warm)
      
        wc_alpha_obs_temp_st = x_guess_temp - x_obs_trajectory[:, jnp.newaxis]
        ws_alpha_obs_temp_st = y_guess_temp - y_obs_trajectory[:, jnp.newaxis]
        ws_beta_obs_temp_st = z_guess_temp - z_obs_trajectory[:, jnp.newaxis]

        wc_alpha_obs_temp_st = wc_alpha_obs_temp_st.transpose(1,0,2)
        ws_alpha_obs_temp_st = ws_alpha_obs_temp_st.transpose(1,0,2)
        ws_beta_obs_temp_st = ws_beta_obs_temp_st.transpose(1,0,2)

        wc_alpha_obs_st = wc_alpha_obs_temp_st.reshape(self.n_agents, self.num*(self.num_obs))
        ws_alpha_obs_st = ws_alpha_obs_temp_st.reshape(self.n_agents, self.num*(self.num_obs))
        ws_beta_obs_st = ws_beta_obs_temp_st.reshape(self.n_agents, self.num*(self.num_obs))

        dist_obs_st = -(wc_alpha_obs_st**2/(self.a_obs )**2) - (ws_alpha_obs_st**2/(self.b_obs )**2)- (ws_beta_obs_st**2/(self.c_obs )**2)+1
        ######################
        b_eq_x = jnp.vstack((x_init_vec, vx_init_vec, ax_init_vec, x_fin_vec )).T
        b_eq_y = jnp.vstack((y_init_vec, vy_init_vec, ay_init_vec, y_fin_vec )).T
        b_eq_z = jnp.vstack((z_init_vec, vz_init_vec, az_init_vec, z_fin_vec )).T


        A_eq = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0], self.P_jax[-1]  ))
        cost_mat = jnp.vstack((  jnp.hstack(( jnp.dot(self.Pddot_jax.T, self.Pddot_jax) + jnp.dot(self.P_jax.T, self.P_jax), A_eq.T )), jnp.hstack(( A_eq, jnp.zeros(( jnp.shape(A_eq)[0], jnp.shape(A_eq)[0] )) )) ))
        cost_mat_inv = jnp.linalg.inv(cost_mat)

        lincost_x = -jnp.hstack((  jnp.dot(self.P_jax.T, x_guess_warm.T).T, b_eq_x ))
        lincost_y = -jnp.hstack((  jnp.dot(self.P_jax.T, y_guess_warm.T).T, b_eq_y ))
        lincost_z = -jnp.hstack((  jnp.dot(self.P_jax.T, z_guess_warm.T).T, b_eq_z ))

        sol_x = jnp.dot(cost_mat_inv, -lincost_x.T).T
        sol_y = jnp.dot(cost_mat_inv, -lincost_y.T).T
        sol_z = jnp.dot(cost_mat_inv, -lincost_z.T).T

        sol_x_bar = sol_x[:,0:self.nvar]
        sol_y_bar = sol_y[:,0:self.nvar]
        sol_z_bar = sol_z[:,0:self.nvar]

        x_guess_warm = jnp.dot( self.P_jax, sol_x_bar.T).T
        y_guess_warm = jnp.dot( self.P_jax, sol_y_bar.T).T
        z_guess_warm = jnp.dot( self.P_jax, sol_z_bar.T).T

        wc_alpha_obs_temp = x_guess_temp[:, jnp.newaxis, :] - x_obs_traj_agent
        ws_alpha_obs_temp = y_guess_temp[:, jnp.newaxis, :] - y_obs_traj_agent
        ws_beta_obs_temp = z_guess_temp[:, jnp.newaxis, :] - z_obs_traj_agent

        wc_alpha_obs = wc_alpha_obs_temp.reshape(self.n_agents,  self.num*(self.num_closest))
        ws_alpha_obs = ws_alpha_obs_temp.reshape(self.n_agents,  self.num*(self.num_closest))
        ws_beta_obs = ws_beta_obs_temp.reshape(self.n_agents,  self.num*(self.num_closest))

        dist_obs_agent = -(wc_alpha_obs**2/(self.a_obs_agent )**2) - (ws_alpha_obs**2/(self.b_obs_agent )**2) -(ws_beta_obs**2/(self.c_obs_agent )**2)+1

        dist_obs = jnp.hstack((dist_obs_st, dist_obs_agent))

        ###############################

        wc_alpha_obs_temp_st_warm = x_guess_warm - x_obs_trajectory[:, jnp.newaxis]
        ws_alpha_obs_temp_st_warm = y_guess_warm - y_obs_trajectory[:, jnp.newaxis]
        ws_beta_obs_temp_st_warm = z_guess_warm - z_obs_trajectory[:, jnp.newaxis]

        wc_alpha_obs_temp_st_warm = wc_alpha_obs_temp_st_warm.transpose(1,0,2)
        ws_alpha_obs_temp_st_warm = ws_alpha_obs_temp_st_warm.transpose(1,0,2)
        ws_beta_obs_temp_st_warm = ws_beta_obs_temp_st_warm.transpose(1,0,2)

        wc_alpha_obs_st_warm = wc_alpha_obs_temp_st_warm.reshape(self.n_agents, self.num*(self.num_obs))
        ws_alpha_obs_st_warm = ws_alpha_obs_temp_st_warm.reshape(self.n_agents, self.num*(self.num_obs))
        ws_beta_obs_st_warm = ws_beta_obs_temp_st_warm.reshape(self.n_agents, self.num*(self.num_obs))


        dist_obs_st_warm = -(wc_alpha_obs_st_warm**2/(self.a_obs )**2) - (ws_alpha_obs_st_warm**2/(self.b_obs )**2) - (ws_beta_obs_st_warm**2/(self.c_obs )**2)+1
        ######################
 
        wc_alpha_obs_temp_warm = x_guess_warm[:, jnp.newaxis, :] - x_obs_traj_agent_warm
        ws_alpha_obs_temp_warm = y_guess_warm[:, jnp.newaxis, :] - y_obs_traj_agent_warm
        ws_beta_obs_temp_warm = z_guess_warm[:, jnp.newaxis, :] - z_obs_traj_agent_warm

        wc_alpha_obs_warm = wc_alpha_obs_temp_warm.reshape(self.n_agents,  self.num*(self.num_closest))
        ws_alpha_obs_warm = ws_alpha_obs_temp_warm.reshape(self.n_agents,  self.num*(self.num_closest))
        ws_beta_obs_warm = ws_beta_obs_temp_warm.reshape(self.n_agents,  self.num*(self.num_closest))

        dist_obs_agent_warm = -(wc_alpha_obs_warm**2/(self.a_obs_agent )**2) - (ws_alpha_obs_warm**2/(self.b_obs_agent )**2) - (ws_beta_obs_warm**2/(self.c_obs_agent )**2)+1

        dist_obs_warm = jnp.hstack((dist_obs_st_warm, dist_obs_agent_warm))

        cost_obs_penalty = jnp.sum(jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.n_agents, self.num*( self.num_closest + self.num_obs) )), dist_obs), axis = 1), axis = 0)
        cost_obs_penalty_warm = jnp.sum(jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.n_agents, self.num*( self.num_closest + self.num_obs) )), dist_obs_warm), axis = 1), axis = 0)

        # print(cost_obs_penalty, cost_obs_penalty_warm, "cost")
        idx = jnp.argmin(jnp.hstack(( cost_obs_penalty, cost_obs_penalty_warm + 3)))

        x = jnp.concatenate( [x_guess_temp.reshape(1, self.n_agents, self.num), x_guess_warm.reshape(1, self.n_agents, self.num)], axis = 0)
        y = jnp.concatenate( [y_guess_temp.reshape(1, self.n_agents, self.num), y_guess_warm.reshape(1, self.n_agents, self.num)], axis = 0)
        z = jnp.concatenate( [z_guess_temp.reshape(1, self.n_agents, self.num), z_guess_warm.reshape(1, self.n_agents, self.num)], axis=0)

        x_guess = x[idx, :, :].reshape(self.n_agents, self.num)
        y_guess = y[idx, :, :].reshape(self.n_agents, self.num)
        z_guess = z[idx, :, :].reshape(self.n_agents, self.num)

        sol_x_bar, sol_y_bar, sol_z_bar, x_guess, y_guess, z_guess, xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess = self.compute_guess(x_guess, y_guess, z_guess)

        x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent = self.compute_obs_dy_traj_prediction(x_guess, y_guess, z_guess)
      
        # plt.figure(2)
        
        # plt.plot(np.array(x_guess_temp[0,  :]).T, np.array(y_guess_temp[0,  :]).T, color = "blue")
        # plt.plot(np.array(x_guess_temp[1,  :]).T, np.array(y_guess_temp[1,  :]).T, color = "green")
        # plt.plot(np.array(x_guess_temp[2,  :]).T, np.array(y_guess_temp[2,  :]).T, color = "yellow")
        # plt.plot(np.array(x_guess_temp[3,  :]).T, np.array(y_guess_temp[3,  :]).T, color = "brown")
        # plt.plot(np.array(x_guess_temp[4,  :]).T, np.array(y_guess_temp[4,  :]).T, color = "blue")
        # plt.plot(np.array(x_guess_temp[5,  :]).T, np.array(y_guess_temp[5,  :]).T, color = "green")
        # plt.plot(np.array(x_guess_temp[6,  :]).T, np.array(y_guess_temp[6,  :]).T, color = "yellow")
        # plt.plot(np.array(x_guess_temp[7,  :]).T, np.array(y_guess_temp[7,  :]).T, color = "brown")

        # plt.show()
       


        return sol_x_bar, sol_y_bar, sol_z_bar, x_guess, y_guess, z_guess, xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess, x_fin_vec, y_fin_vec, z_fin_vec, x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent

        
    
    #######################################
    @partial(jit, static_argnums=(0),backend="gpu")	
    def compute_boundary_vec(self, initial_state_x, initial_state_y, initial_state_z, x_fin, y_fin, z_fin):

        n_agents = jnp.shape(initial_state_x)[0]

        x_init_vec = initial_state_x[:, 0].reshape(n_agents, 1)
        y_init_vec = initial_state_y[:, 0].reshape(n_agents, 1)
        z_init_vec = initial_state_z[:, 0].reshape(n_agents, 1)

        vx_init_vec = initial_state_x[:, 1].reshape(n_agents, 1)
        vy_init_vec = initial_state_y[:, 1].reshape(n_agents, 1)
        vz_init_vec = initial_state_z[:, 1].reshape(n_agents, 1)

        ax_init_vec = initial_state_x[:, 2].reshape(n_agents, 1)
        ay_init_vec = initial_state_y[:, 2].reshape(n_agents, 1)
        az_init_vec = initial_state_z[:, 2].reshape(n_agents, 1)

        x_fin_vec = x_fin.reshape(n_agents, 1)
        y_fin_vec = y_fin.reshape(n_agents, 1)
        z_fin_vec = z_fin.reshape(n_agents, 1)

        b_eq_x = jnp.hstack((x_init_vec, vx_init_vec, ax_init_vec, x_fin_vec))
        b_eq_y = jnp.hstack((y_init_vec, vy_init_vec, ay_init_vec, y_fin_vec))
        b_eq_z = jnp.hstack((z_init_vec, vz_init_vec, az_init_vec, z_fin_vec))

		
        return b_eq_x, b_eq_y, b_eq_z
    
    #####################################################
    @partial(jit, static_argnums=(0, ))
    def compute_v(self, xdot_guess, ydot_guess, zdot_guess):

        wc_alpha_vx = xdot_guess
        ws_alpha_vy = ydot_guess

        alpha_v = jnp.arctan2( ws_alpha_vy, wc_alpha_vx)	

        wc_beta_v = zdot_guess
        ws_beta_v = wc_alpha_vx/jnp.cos(alpha_v)
        beta_v = jnp.arctan2( ws_beta_v, wc_beta_v)

        c1_d_v = 1.0*self.rho_ineq
        c2_d_v = 1.0*self.rho_ineq*(wc_alpha_vx*jnp.cos(alpha_v)*jnp.sin(beta_v) + ws_alpha_vy*jnp.sin(alpha_v)*jnp.sin(beta_v) +wc_beta_v *jnp.cos(beta_v) )

        d_v = c2_d_v/c1_d_v

        d_v = jnp.minimum(self.v_max*jnp.ones((self.n_agents, self.num)), d_v   )

        res_vx_vec = xdot_guess-d_v*jnp.cos(alpha_v) *jnp.sin(beta_v)
        res_vy_vec = ydot_guess-d_v*jnp.sin(alpha_v) *jnp.sin(beta_v)
        res_vz_vec = zdot_guess-d_v*jnp.cos(beta_v)


        res_vel_vec = jnp.hstack(( res_vx_vec, res_vy_vec , res_vz_vec  ))

        return alpha_v, beta_v, d_v, res_vx_vec, res_vy_vec, res_vz_vec, res_vel_vec

    ##########################
    @partial(jit, static_argnums=(0, ))
    def compute_acc(self, xddot_guess, yddot_guess, zddot_guess):

        wc_alpha_ax = xddot_guess
        ws_alpha_ay = yddot_guess

        alpha_a = jnp.arctan2( ws_alpha_ay, wc_alpha_ax)	

        wc_beta_a = zddot_guess
        ws_beta_a = wc_alpha_ax/jnp.cos(alpha_a)

        beta_a = jnp.arctan2( ws_beta_a, wc_beta_a)

        c1_d_a = 1.0*self.rho_ineq
        c2_d_a = 1.0*self.rho_ineq*(wc_alpha_ax*jnp.cos(alpha_a)*jnp.sin(beta_a) + ws_alpha_ay*jnp.sin(alpha_a)*jnp.sin(beta_a) +wc_beta_a *jnp.cos(beta_a) )

        d_a = c2_d_a/c1_d_a

        d_a = jnp.minimum(self.a_max*jnp.ones((self.n_agents, self.num)), d_a   )

        res_ax_vec = xddot_guess-d_a*jnp.cos(alpha_a) *jnp.sin(beta_a)
        res_ay_vec = yddot_guess-d_a*jnp.sin(alpha_a) *jnp.sin(beta_a)
        res_az_vec = zddot_guess-d_a*jnp.cos(beta_a)

        res_acc_vec = jnp.hstack(( res_ax_vec,  res_ay_vec , res_az_vec ))

        return alpha_a, beta_a, d_a, res_ax_vec, res_ay_vec, res_az_vec, res_acc_vec

    ##################################
    @partial(jit, static_argnums=(0, ))	
    def initial_alpha_d(self, x_fin, y_fin, z_fin, x_agent_guess, y_agent_guess, z_agent_guess, xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess, x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x_obs_dynamic_trajectory, y_obs_dynamic_trajectory, z_obs_dynamic_trajectory, x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent, lamda_x, lamda_y, lamda_z  )  :    
        
        alpha_obs_st, beta_obs_st, d_obs_st,  wc_alpha_obs_st, ws_alpha_obs_st, wc_beta_obs_st, ws_beta_obs_st, res_x_obs_vec_st, res_y_obs_vec_st, res_z_obs_vec_st = self.compute_alpha_st( x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x_agent_guess, y_agent_guess, z_agent_guess  )
        
        alpha_obs_dy, beta_obs_dy, d_obs_dy, wc_alpha_obs_dy, ws_alpha_obs_dy, wc_beta_obs_dy, ws_beta_obs_dy, res_x_obs_vec_dy, res_y_obs_vec_dy, res_z_obs_vec_dy  = self.compute_alpha_dy( x_obs_dynamic_trajectory, y_obs_dynamic_trajectory, z_obs_dynamic_trajectory, x_agent_guess, y_agent_guess, z_agent_guess  )

        alpha_obs_agent, beta_obs_agent, d_obs_agent, wc_alpha_obs_agent, ws_alpha_obs_agent, wc_beta_obs_agent, ws_beta_obs_agent, res_x_obs_vec_agent, res_y_obs_vec_agent, res_z_obs_vec_agent  = self.compute_alpha_agent(x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent, x_agent_guess, y_agent_guess, z_agent_guess  )

        alpha_obs = jnp.hstack((alpha_obs_st, alpha_obs_dy, alpha_obs_agent))
        beta_obs = jnp.hstack((beta_obs_st, beta_obs_dy, beta_obs_agent ))
        d_obs = jnp.hstack((d_obs_st, d_obs_dy, d_obs_agent))

        res_x_obs_vec = jnp.hstack((res_x_obs_vec_st, res_x_obs_vec_dy, res_x_obs_vec_agent))
        res_y_obs_vec = jnp.hstack((res_y_obs_vec_st, res_y_obs_vec_dy, res_y_obs_vec_agent))
        res_z_obs_vec = jnp.hstack((res_z_obs_vec_st, res_z_obs_vec_dy, res_z_obs_vec_agent))

        res_obs_vec = jnp.hstack((res_x_obs_vec, res_y_obs_vec, res_z_obs_vec))
        
        #########Velocity Term

        alpha_v, beta_v, d_v, res_vx_vec, res_vy_vec, res_vz_vec, res_vel_vec = self.compute_v(xdot_guess, ydot_guess, zdot_guess)

        ################# acceleration terms

        alpha_a, beta_a, d_a, res_ax_vec, res_ay_vec, res_az_vec, res_acc_vec = self.compute_acc(xddot_guess, yddot_guess, zddot_guess)

        ###########################################################

        lamda_x, lamda_y, lamda_z, res_norm = self.compute_lamda(res_vx_vec, res_vy_vec, res_vz_vec, res_ax_vec, res_ay_vec, res_az_vec,res_x_obs_vec, res_y_obs_vec, res_z_obs_vec,lamda_x, lamda_y, lamda_z,res_vel_vec, res_acc_vec, res_obs_vec)

        return alpha_obs, beta_obs, d_obs, alpha_v, beta_v,  d_v, alpha_a, beta_a,  d_a, lamda_x,lamda_y,lamda_z
    
    
    ##############################3
    @partial(jit, static_argnums=(0, ))	
    def compute_agents(self, x_fin, y_fin, z_fin, x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x_obs_dynamic_trajectory, y_obs_dynamic_trajectory, z_obs_dynamic_trajectory, x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent, d_obs, alpha_obs, beta_obs, d_v, alpha_v, beta_v, d_a, alpha_a, beta_a, lamda_x, lamda_y, lamda_z, b_eq_x, b_eq_y, b_eq_z, sol_x_bar, sol_y_bar, sol_z_bar):

        b_projection_x = sol_x_bar
        b_projection_y = sol_y_bar
        b_projection_z = sol_z_bar

        b_obs_x, b_obs_y, b_obs_z = self.compute_b_part(sol_x_bar, sol_y_bar, sol_z_bar, d_obs, alpha_obs, beta_obs, x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x_obs_dynamic_trajectory, y_obs_dynamic_trajectory, z_obs_dynamic_trajectory, x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent)
        
    
        b_ax_ineq = d_a*jnp.cos(alpha_a)* jnp.sin(beta_a)
        b_ay_ineq = d_a*jnp.sin(alpha_a)* jnp.sin(beta_a)
        b_az_ineq = d_a*jnp.cos(beta_a)

        b_vx_ineq = d_v*jnp.cos(alpha_v) * jnp.sin(beta_v)
        b_vy_ineq = d_v*jnp.sin(alpha_v) * jnp.sin(beta_v)
        b_vz_ineq = d_v*jnp.cos(beta_v)

        lamda_x = lamda_x 
        lamda_y = lamda_y 
        lamda_z = lamda_z 

        primal_sol_x, primal_sol_y, primal_sol_z, x, xdot, xddot, y, ydot, yddot, z, zdot, zddot = self.compute_x_batch(b_eq_x, b_eq_y, b_eq_z, b_projection_x, b_projection_y, b_projection_z, b_obs_x, b_obs_y, b_obs_z, b_ax_ineq, b_ay_ineq, b_az_ineq, b_vx_ineq, b_vy_ineq, b_vz_ineq, lamda_x, lamda_y, lamda_z)        
        
        return primal_sol_x, primal_sol_y, primal_sol_z, x, xdot, xddot, y, ydot, yddot, z, zdot, zddot
    ##################################################
    @partial(jit, static_argnums=(0, ))	
    def alpha_d(self, x, y, z, xdot, ydot, zdot, xddot, yddot, zddot, x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x_obs_dynamic_trajectory, y_obs_dynamic_trajectory, z_obs_dynamic_trajectory, x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent, lamda_x, lamda_y, lamda_z ):
        
        alpha_obs_st, beta_obs_st, d_obs_st,  wc_alpha_obs_st, ws_alpha_obs_st, wc_beta_obs_st, ws_beta_obs_st, res_x_obs_vec_st, res_y_obs_vec_st, res_z_obs_vec_st = self.compute_alpha_st( x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x, y , z)
        
        alpha_obs_dy, beta_obs_dy, d_obs_dy, wc_alpha_obs_dy, ws_alpha_obs_dy, wc_beta_obs_dy, ws_beta_obs_dy, res_x_obs_vec_dy, res_y_obs_vec_dy, res_z_obs_vec_dy = self.compute_alpha_dy( x_obs_dynamic_trajectory, y_obs_dynamic_trajectory, z_obs_dynamic_trajectory, x, y, z  )

        alpha_obs_agent, beta_obs_agent, d_obs_agent, wc_alpha_obs_agent, ws_alpha_obs_agent, wc_beta_obs_agent, ws_beta_obs_agent, res_x_obs_vec_agent, res_y_obs_vec_agent, res_z_obs_vec_agent = self.compute_alpha_agent(x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent, x, y, z  )

        alpha_obs = jnp.hstack((alpha_obs_st, alpha_obs_dy, alpha_obs_agent))
        beta_obs = jnp.hstack((beta_obs_st, beta_obs_dy, beta_obs_agent ))

        d_obs = jnp.hstack((d_obs_st, d_obs_dy, d_obs_agent))

        res_x_obs_vec = jnp.hstack((res_x_obs_vec_st, res_x_obs_vec_dy, res_x_obs_vec_agent)) 
        res_y_obs_vec = jnp.hstack((res_y_obs_vec_st, res_y_obs_vec_dy, res_y_obs_vec_agent)) 
        res_z_obs_vec = jnp.hstack((res_z_obs_vec_st, res_z_obs_vec_dy, res_z_obs_vec_agent))


        res_obs_vec = jnp.hstack((res_x_obs_vec, res_y_obs_vec, res_z_obs_vec)) 
    
        ##############################

        alpha_v, beta_v, d_v, res_vx_vec, res_vy_vec, res_vz_vec, res_vel_vec = self.compute_v(xdot, ydot, zdot)

        ################# acceleration terms

        alpha_a, beta_a, d_a, res_ax_vec, res_ay_vec, res_az_vec, res_acc_vec = self.compute_acc(xddot, yddot, zddot)

        #####################################
        
        lamda_x, lamda_y, lamda_z, res_norm = self.compute_lamda(res_vx_vec, res_vy_vec, res_vz_vec, res_ax_vec, res_ay_vec, res_az_vec,res_x_obs_vec, res_y_obs_vec, res_z_obs_vec,lamda_x, lamda_y, lamda_z,res_vel_vec, res_acc_vec, res_obs_vec)

        res_norm_obs = jnp.linalg.norm(res_obs_vec, axis = 1)
        res_norm_acc  = jnp.linalg.norm(res_acc_vec, axis =1)
        res_norm_vel = jnp.linalg.norm(res_vel_vec, axis =1)
        
        return alpha_obs, beta_obs, d_obs, alpha_v, beta_v, d_v, alpha_a, beta_a, d_a, lamda_x,lamda_y, lamda_z, res_norm, res_norm_obs, res_norm_acc, res_norm_vel,
                
               
    ########################################
    @partial(jit, static_argnums=(0, ))	
    def compute_variable(self,  initial_state_x, initial_state_y, initial_state_z,  x_fin, y_fin, z_fin, lamda_x, lamda_y, lamda_z, x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x_obs_dynamic_trajectory, y_obs_dynamic_trajectory, z_obs_dynamic_trajectory, x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent, sol_x_bar, sol_y_bar, sol_z_bar, x_agent_guess, y_agent_guess, z_agent_guess, xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess ):

        x_guess = x_agent_guess
        y_guess = y_agent_guess
        z_guess = z_agent_guess

        b_eq_x, b_eq_y, b_eq_z = self.compute_boundary_vec( initial_state_x.T, initial_state_y.T, initial_state_z.T, x_fin, y_fin, z_fin) 
        alpha_obs_init, beta_obs_init, d_obs_init, alpha_v_init, beta_v_init, d_v_init, alpha_a_init, beta_a_init, d_a_init, lamda_x_init, lamda_y_init, lamda_z_init = self.initial_alpha_d( x_fin, y_fin, z_fin, x_agent_guess, y_agent_guess, z_agent_guess, xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess, x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x_obs_dynamic_trajectory, y_obs_dynamic_trajectory, z_obs_dynamic_trajectory, x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent, lamda_x, lamda_y, lamda_z)
    
        c_x_init = jnp.zeros((self.n_agents,self.nvar))
        c_y_init = jnp.zeros((self.n_agents,self.nvar))
        c_z_init = jnp.zeros((self.n_agents,self.nvar))

        x_init = jnp.zeros((self.n_agents,self.num))
        xdot_init = jnp.zeros((self.n_agents,self.num))
        xddot_init = jnp.zeros((self.n_agents,self.num))

        y_init = jnp.zeros((self.n_agents,self.num))
        ydot_init = jnp.zeros((self.n_agents,self.num))
        yddot_init = jnp.zeros((self.n_agents,self.num))

        z_init = jnp.zeros((self.n_agents,self.num))
        zdot_init = jnp.zeros((self.n_agents,self.num))
        zddot_init = jnp.zeros((self.n_agents,self.num))

        res_norm_init = jnp.zeros(self.n_agents)
        res_norm_acc_init = jnp.zeros(self.n_agents)
        res_norm_vel_init = jnp.zeros(self.n_agents)
        res_norm_obs_init = jnp.zeros(self.n_agents)

        x_obs_traj_agent_init = x_obs_traj_agent
        y_obs_traj_agent_init = y_obs_traj_agent
        z_obs_traj_agent_init = z_obs_traj_agent


        def lax_projection(carry, proj_iter ):

            c_x, c_y, c_z, x, xdot, xddot, y, ydot, yddot, z, zdot, zddot, alpha_obs, beta_obs, d_obs, alpha_v, beta_v, d_v, alpha_a, beta_a, d_a, lamda_x, lamda_y, lamda_z, res_norm, res_norm_obs, res_norm_acc, res_norm_vel, x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent = carry

            c_x, c_y, c_z, x, xdot, xddot, y, ydot, yddot, z, zdot, zddot = self.compute_agents(x_fin, y_fin, z_fin, x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x_obs_dynamic_trajectory, y_obs_dynamic_trajectory, z_obs_dynamic_trajectory, x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent, d_obs, alpha_obs, beta_obs, d_v, alpha_v, beta_v, d_a, alpha_a, beta_a,  lamda_x, lamda_y, lamda_z, b_eq_x, b_eq_y, b_eq_z, sol_x_bar, sol_y_bar, sol_z_bar )
            
            alpha_obs, beta_obs, d_obs, alpha_v, beta_v, d_v, alpha_a, beta_a, d_a, lamda_x, lamda_y, lamda_z, res_norm, res_norm_obs, res_norm_acc, res_norm_vel = self.alpha_d(  x, y, z, xdot, ydot, zdot, xddot, yddot, zddot, x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x_obs_dynamic_trajectory, y_obs_dynamic_trajectory, z_obs_dynamic_trajectory, x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent, lamda_x, lamda_y, lamda_z)

            
            # x_obs_traj_agent, y_obs_traj_agent = self.compute_obs_dy_traj_prediction( x, y)
            
            return (c_x, c_y, c_z, x, xdot, xddot, y, ydot, yddot, z, zdot, zddot, alpha_obs, beta_obs, d_obs, alpha_v, beta_v, d_v, alpha_a, beta_a, d_a, lamda_x, lamda_y, lamda_z, res_norm, res_norm_obs, res_norm_acc, res_norm_vel, x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent), x
                   
        carry_init = c_x_init, c_y_init, c_z_init, x_init, xdot_init, xddot_init, y_init, ydot_init, yddot_init, z_init, zdot_init, zddot_init, alpha_obs_init, beta_obs_init, d_obs_init, alpha_v_init, beta_v_init, d_v_init, alpha_a_init, beta_a_init, d_a_init, lamda_x_init, lamda_y_init, lamda_z_init, res_norm_init, res_norm_obs_init, res_norm_acc_init, res_norm_vel_init, x_obs_traj_agent_init, y_obs_traj_agent_init, z_obs_traj_agent_init
        carry_fin, result = lax.scan(lax_projection, carry_init, jnp.arange(self.maxiter_proj))
        c_x, c_y, c_z, x, xdot, xddot, y, ydot, yddot, z, zdot, zddot, alpha_obs, beta_obs, d_obs, alpha_v, beta_v, d_v, alpha_a, beta_a, d_a, lamda_x, lamda_y, lamda_z, res_norm, res_norm_obs, res_norm_acc, res_norm_vel, x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent = carry_fin
        


        # d_obs = d_obs_init
        # alpha_obs = alpha_obs_init

        # d_v =  d_v_init
        # alpha_v = alpha_v_init

        # d_a = d_a_init
        # alpha_a = alpha_a_init

        # lamda_x = lamda_x_init
        # lamda_y = lamda_y_init
        
        # res = []
        # res_obs = []
        # res_acc = [] 
        # res_vel =[] 
        # for i in range(0,100):

        #     c_x, c_y, x, xdot, xddot, y, ydot, yddot = self.compute_agents(x_fin, y_fin, x_obs_trajectory, y_obs_trajectory, x_obs_dynamic_trajectory, y_obs_dynamic_trajectory, x_obs_traj_agent1, y_obs_traj_agent1, d_obs, alpha_obs, d_v, alpha_v, d_a, alpha_a,  lamda_x, lamda_y, b_eq_x, b_eq_y, sol_x_bar, sol_y_bar )
            
        #     alpha_obs, d_obs, alpha_v, d_v, alpha_a, d_a, lamda_x, lamda_y,  res_norm, res_norm_obs, res_norm_acc, res_norm_vel = self.alpha_d(  x, y, xdot, ydot, xddot, yddot, x_obs_trajectory, y_obs_trajectory, x_obs_dynamic_trajectory, y_obs_dynamic_trajectory, x_obs_traj_agent1, y_obs_traj_agent1, lamda_x, lamda_y)
           
        #     x_obs_traj_agent, y_obs_traj_agent = self.compute_obs_dy_traj_prediction( x, y)   
    
        #     res.append(res_norm)
        #     res_obs.append(res_norm_obs)
        #     res_acc.append(res_norm_acc)
        #     res_vel.append(res_norm_vel)

    
        
        # th = np.linspace(0, 2*np.pi, self.num)
        # plt.figure(3333)
    
        # plt.plot(np.array(x[0, :]).T, np.array(y[0, :]).T , color = 'blue')
        # plt.plot(np.array(x[1, :]).T, np.array(y[1, :]).T , color = 'green')
        # plt.plot(np.array(x[2, :]).T, np.array(y[2, :]).T , color = 'yellow')
        # plt.plot(np.array(x[3, :]).T, np.array(y[3, :]).T , color = 'brown')    


        # plt.show()



        # # sample_id = 0  # choose which sample you want to visualize
        # res = (np.array(res))#[idx_ellite_projection[0:70],:]
        # res_obs = (np.array(res_obs))#[idx_ellite_projection[0:70],:]
        # res_acc = (np.array(res_acc))#[idx_ellite_projection[0:70],:]
        # res_vel = (np.array(res_vel))#[idx_ellite_projection[0:70],:]
       

        # # print(jnp.shape(res))


        # plt.figure(3)
        # plt.plot( res[:, 0].T, color = "red")
        # plt.plot( res[:, 1].T, color = "blue")
        # plt.plot( res[:, 2].T, color = "green")
        # plt.plot( res[:, 3].T, color = "orange")

        # plt.figure(4)
        # plt.plot( res_obs[:, 0].T, color = "red")
        # plt.plot( res_obs[:, 1].T, color = "blue")
        # plt.plot( res_obs[:, 2].T, color = "green")
        # plt.plot( res_obs[:, 3].T, color = "orange")
        

        # plt.figure(5)
        # plt.plot( res_acc[:, 0].T, color = "red")
        # plt.plot( res_acc[:, 1].T, color = "blue")
        # plt.plot( res_acc[:, 2].T, color = "green")
        # plt.plot( res_acc[:, 3].T, color = "orange")

        # plt.figure(6)
        # plt.plot( res_vel[:, 0].T, color = "red")
        # plt.plot( res_vel[:, 1].T, color = "blue")
        # plt.plot( res_vel[:, 2].T, color = "green")
        # plt.plot( res_vel[:, 3].T, color = "orange")

        
        # plt.show()


        return c_x, c_y, c_z, x, xdot, xddot, y, ydot, yddot, z, zdot, zddot, res_norm, x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent

    
    ###########################################
    @partial(jit, static_argnums=(0, ))
    def compute_controls(self, c_x_best, c_y_best, c_z_best):

        xdot_best = jnp.dot( self.Pdot_jax, c_x_best.T).T
        ydot_best = jnp.dot( self.Pdot_jax, c_y_best.T).T
        zdot_best = jnp.dot( self.Pdot_jax, c_z_best.T).T

        xddot_best = jnp.dot( self.Pddot_jax, c_x_best.T).T
        yddot_best = jnp.dot( self.Pddot_jax, c_y_best.T).T
        zddot_best = jnp.dot( self.Pddot_jax, c_z_best.T).T

        vx_control = jnp.mean(xdot_best[:, 0:12], axis = 1)
        vy_control = jnp.mean(ydot_best[:, 0:12], axis = 1)
        vz_control = jnp.mean(zdot_best[:, 0:12], axis = 1)

        ax_control = jnp.mean(xddot_best[:, 0:12], axis = 1)
        ay_control = jnp.mean(yddot_best[:, 0:12], axis = 1)
        az_control = jnp.mean(zddot_best[:, 0:12], axis = 1)
       
        return vx_control, vy_control, vz_control, ax_control, ay_control, az_control
    
    ########################################

    @partial(jit, static_argnums=(0, ))	
    def Compute_CEM(self, initial_state_x, initial_state_y, initial_state_z, x_fin, y_fin, z_fin, lamda_x, lamda_y, lamda_z, x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x_obs_dynamic_trajectory, y_obs_dynamic_trajectory, z_obs_dynamic_trajectory, x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent, sol_x_bar, sol_y_bar, sol_z_bar, x_agent_guess, y_agent_guess, z_agent_guess, xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess, arc_vec, x_waypoint, y_waypoint, z_waypoint):        
        
        c_x, c_y, c_z, x_agent, xdot_agent, xddot_agent, y_agent, ydot_agent, yddot_agent, z_agent, zdot_agent, zddot_agent, res_norm, x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent = self.compute_variable(initial_state_x, initial_state_y, initial_state_z, x_fin, y_fin, z_fin, lamda_x, lamda_y, lamda_z, x_obs_trajectory, y_obs_trajectory, z_obs_trajectory, x_obs_dynamic_trajectory, y_obs_dynamic_trajectory, z_obs_dynamic_trajectory, x_obs_traj_agent, y_obs_traj_agent, z_obs_traj_agent, sol_x_bar, sol_y_bar, sol_z_bar, x_agent_guess, y_agent_guess, z_agent_guess, xdot_guess, ydot_guess, zdot_guess, xddot_guess, yddot_guess, zddot_guess)

                
        # plt.figure(33)
        # for i in range(200):
        #     plt.plot(np.array(x_agent[0,  :]).T, np.array(y_agent[0,:]).T , color = 'blue')
        #     plt.plot(np.array(x_agent[1,  :]).T, np.array(y_agent[1, :]).T , color = 'green')
        #     plt.plot(np.array(x_agent[2,  :]).T, np.array(y_agent[2, :]).T , color = 'yellow')
        #     plt.plot(np.array(x_agent[3,  :]).T, np.array(y_agent[3, :]).T , color = 'brown')
        # plt.show()

        x_guess_warm = x_agent
        y_guess_warm = y_agent
        z_guess_warm = z_agent


        return c_x, c_y, c_z, x_agent, y_agent, z_agent, xdot_agent, ydot_agent, zdot_agent, xddot_agent, yddot_agent, zddot_agent, x_guess_warm, y_guess_warm, z_guess_warm
