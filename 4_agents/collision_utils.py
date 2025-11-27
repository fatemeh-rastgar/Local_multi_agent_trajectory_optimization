import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt




def compute_obs_dy_traj_prediction(x_agent, y_agent, z_agent, n_agents):
    J = jnp.arange(n_agents - 1)                   # (n_agents-1,)
    I = jnp.arange(n_agents).reshape(-1, 1)        # (n_agents, 1)
    other_idx = J + (J >= I).astype(J.dtype)       # (n_agents, n_agents-1)

    x_fixed, y_fixed, z_fixed = 0.0 * x_agent, 0.0 * y_agent, 0.0 * z_agent
    m = jnp.ones((n_agents, 1), dtype=bool)

    x_blend, y_blend, z_blend = jnp.where(m, x_agent, x_fixed), jnp.where(m, y_agent, y_fixed), jnp.where(m, z_agent, z_fixed)
    build_obs = lambda data, other_idx: jnp.take(data, other_idx, axis=0)

    x_obs_dy_trajectory, y_obs_dy_trajectory, z_obs_dy_trajectory = build_obs(x_blend, other_idx), build_obs(y_blend, other_idx), build_obs(z_blend, other_idx)
    return x_obs_dy_trajectory, y_obs_dy_trajectory, z_obs_dy_trajectory


def compute_no_collisions(x, y, z, x_obs_st_trajectory, y_obs_st_trajectory, z_obs_st_trajectory,
                          n_agents, num_obs,
                          a_obs_agent, b_obs_agent, c_obs_agent,
                          a_obs, b_obs, c_obs):

    # Transpose to match existing code logic
    x, y, z = x.T, y.T, z.T
    x_obs_st_trajectory, y_obs_st_trajectory, z_obs_st_trajectory = x_obs_st_trajectory[:, 0], y_obs_st_trajectory[:, 0], z_obs_st_trajectory[:, 0]
    shape_x = jnp.shape(x)[1]

    # --- dynamic obstacles (agents treated as moving obstacles to each other) ---
    x_obs_trajectory, y_obs_trajectory, z_obs_trajectory = compute_obs_dy_traj_prediction(x, y, z, n_agents)

    # --- agent-agent collisions ---
    wc_alpha_obs = (x[:, jnp.newaxis, :] - x_obs_trajectory).reshape(n_agents, shape_x*(n_agents-1))
    ws_alpha_obs = (y[:, jnp.newaxis, :] - y_obs_trajectory).reshape(n_agents, shape_x*(n_agents-1))
    wt_alpha_obs = (z[:, jnp.newaxis, :] - z_obs_trajectory).reshape(n_agents, shape_x*(n_agents-1))

    dist_agent = -(wc_alpha_obs**2 / (a_obs_agent - 0.04)**2) \
                 - (ws_alpha_obs**2 / (b_obs_agent - 0.04)**2) \
                 - (wt_alpha_obs**2 / (c_obs_agent - 0.04)**2) + 1

    dist_agent1 = +(wc_alpha_obs**2 ) \
                 + (ws_alpha_obs**2 ) \
                 + (wt_alpha_obs**2 ) 

    print(np.shape(dist_agent1), "dist_agent1")
    mean_agent = jnp.max(dist_agent.reshape(n_agents, n_agents-1, shape_x), axis=(0,1))  # (shape_x,)
    dist_agent = jnp.maximum(0, dist_agent)
    

    NO_agents_coll = np.count_nonzero(dist_agent)
    print("Agent collisions:", NO_agents_coll)

    # --- agent-static collisions ---
    wc_alpha_st = (x - x_obs_st_trajectory[:, None, None]).reshape(n_agents, shape_x*num_obs)
    ws_alpha_st = (y - y_obs_st_trajectory[:, None, None]).reshape(n_agents, shape_x*num_obs)
    wt_alpha_st = (z - z_obs_st_trajectory[:, None, None]).reshape(n_agents, shape_x*num_obs)

    dist_st = -(wc_alpha_st**2 / (a_obs - 0.04)**2) \
              - (ws_alpha_st**2 / (b_obs - 0.04)**2) \
              - (wt_alpha_st**2 / (c_obs - 0.04)**2) + 1

    dist_st1 = +(wc_alpha_st**2 ) \
               + (ws_alpha_st**2 ) \
              +(wt_alpha_st**2 ) 

    mean_static = jnp.max(dist_st.reshape(n_agents, num_obs, shape_x), axis=(0,1))  # (shape_x,)

    dist_st = jnp.maximum(0, dist_st)

    
    NO_static_coll = np.count_nonzero(dist_st)
    print("Static collisions:", NO_static_coll)

    NO_dynamic_coll = 0  # Optional extension for dynamic environment

    # --- Plot results ---
    plt.figure()
    plt.plot(mean_agent, linewidth=3, label=" agent-agent ")
    plt.plot(mean_static, linewidth=3, label=" agent-static obstacle")
    plt.plot(0*np.ones(473), linewidth=3, label="Satisfaction boundary")
    plt.xlabel("MPC iteration",fontsize=14, labelpad=10)
    plt.ylabel("Value of collision avoidance constraints",fontsize=14, labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return NO_agents_coll, NO_static_coll, NO_dynamic_coll
