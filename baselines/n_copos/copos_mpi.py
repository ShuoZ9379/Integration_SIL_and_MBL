from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common import colorize
from mpi4py import MPI
from collections import deque
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from contextlib import contextmanager

from baselines.n_copos.eta_omega_dual import EtaOmegaOptimizer

def traj_segment_generator(pi, n_active_policy, pi_vf, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    n_policy = len(pi)

    pi_ids = np.zeros(horizon, 'int32')
    logprobs = np.zeros([horizon, n_policy], 'float32')
    logprob = np.zeros(n_policy, 'float32')

    while True:
        i = t % horizon
        pi_id = n_policy - n_active_policy + (i % n_active_policy)
        prevac = ac
        ac, _ = pi[pi_id].act(stochastic, ob)
        _, vpred = pi_vf.act(stochastic, ob)

        for j in range(n_policy):
            logprob[j] = pi[j].logprob(ac, ob)

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                   "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets" : ep_rets, "ep_lens" : ep_lens, "logprobs" : logprobs,
                   "pi_ids" : pi_ids}
            _, vpred = pi_vf.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        pi_ids[i] = pi_id
        logprobs[i,:] = logprob
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def add_vtarg_and_adv_retrace(seg, gamma, lam, act_pi_ids):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])

    n_steps = len(seg["vpred"])

    pi_ids = seg["pi_ids"]
    logprobs = seg["logprobs"]
    old_logprob = logprobs[range(logprobs.shape[0]), pi_ids]
    act_logprob = logprobs[range(logprobs.shape[0]), act_pi_ids]
    prob_ratio = np.minimum(1., np.exp(act_logprob - old_logprob))

    rew = seg["rew"]
    seg["adv"] = gen_adv = np.empty_like(seg["vpred"])
    for rev_k in range(n_steps):
        k = n_steps - rev_k - 1
        if new[k] or rev_k == 0:  # this is a new path. always true for rev_k == 0
            gen_adv[k] = prob_ratio[k] * (rew[k] - vpred[k])
        else:
            gen_adv[k] = prob_ratio[k] * (rew[k] + gamma * vpred[k + 1] - vpred[k] + gamma * lam * gen_adv[k + 1])

    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def get_targets(self, obs, rwd, done, discount, lam, prob_ratio=None):
    # computes v_update targets
    v_values = self.vbase.get_v(obs)
    gen_adv = np.empty_like(v_values)
    if prob_ratio is None:
        prob_ratio = np.ones([len(v_values)])
    for rev_k, v in enumerate(reversed(v_values)):
        k = len(v_values) - rev_k - 1
        if done[k]:  # this is a new path. always true for rev_k == 0
            gen_adv[k] = prob_ratio[k] * (rwd[k] - v_values[k])
        else:
            gen_adv[k] = prob_ratio[k] * (rwd[k] + discount * v_values[k + 1] - v_values[k] + discount * lam * gen_adv[k + 1])
    return gen_adv + v_values, gen_adv

# It returns target v-values (if you want to update the v-function) and generalized advantage values 
# (for policy update). Function should be called with this as the proba_ratio

# retrace_proba_ratio = np.minimum(1., np.exp(act_logprob - old_logprob))

# where act_logprob is the log prob. of current policy and old_logprob is the log prob. 
# of whatever policy generated the data.

# All inputs are matrices. Several trajectories can be stacked. done[i] is True if a terminal state is reached. 
# It is assumed that done[-1] is True (need to change the code otherwise as gen_adv would not be properly initialized).


def split_traj_segment(pis, seg):
    """
    Split trajectory segment among different policies
    :param pis:
    :param seg:
    :return:
    """

    seg_split = []
    for i in range(len(pis)):
        d = {}
        for key in seg:
            if isinstance(seg[key], float):
                d[key] = seg[key]
            else:
                d[key] = seg[key][i::len(pis)]
        seg_split.append(d)

        # seg_split.append({"ob" : seg["ob"][i::len(pis)],
        #                   "ac": seg["ac"][i::len(pis)],
        #                   "adv": seg["adv"][i::len(pis)],
        #                   "tdlamret": seg["tdlamret"][i::len(pis)],
        #                   "vpred": seg["vpred"][i::len(pis)]})

    return seg_split

def eta_search(w_theta, w_beta, eta, omega, allmean, compute_losses, get_flat, set_from_flat, pi, epsilon, args):
    """
    Binary search for eta for finding both valid log-linear "theta" and non-linear "beta" parameter values
    :return: new eta
    """

    w_theta = w_theta.reshape(-1,)
    w_beta = w_beta.reshape(-1,)
    all_params = get_flat()
    best_params = all_params
    param_theta, param_beta = pi.all_to_theta_beta(all_params)
    prev_param_theta = np.copy(param_theta)
    prev_param_beta = np.copy(param_beta)
    final_gain = -1e20
    final_constraint_val = float('nan')
    gain_before, kl, *_ = allmean(np.array(compute_losses(*args)))

    min_ratio = 0.1
    max_ratio = 10
    ratio = max_ratio

    for _ in range(10):
        cur_eta = ratio * eta
        cur_param_theta = (cur_eta * prev_param_theta + w_theta) / (cur_eta + omega)
        cur_param_beta = prev_param_beta + w_beta / cur_eta

        thnew = pi.theta_beta_to_all(cur_param_theta, cur_param_beta)
        set_from_flat(thnew)

        # TEST
        if np.min(np.real(np.linalg.eigvals(pi.get_prec_matrix()))) < 0:
            print("Negative definite covariance!")

        if np.min(np.imag(np.linalg.eigvals(pi.get_prec_matrix()))) != 0:
            print("Covariance has imaginary eigenvalues")

        gain, kl, *_ = allmean(np.array(compute_losses(*args)))

        # TEST
        # print(ratio, gain, kl)

        if all((not np.isnan(kl), kl <= epsilon)):
            if all((not np.isnan(gain), gain > final_gain)):
                eta = cur_eta
                final_gain = gain
                final_constraint_val = kl
                best_params = thnew

            max_ratio = ratio
            ratio = 0.5 * (max_ratio + min_ratio)
        else:
            min_ratio = ratio
            ratio = 0.5 * (max_ratio + min_ratio)

    if any((np.isnan(final_gain), np.isnan(final_constraint_val), final_constraint_val >= epsilon)):
        logger.log("eta_search: Line search condition violated. Rejecting the step!")
        if np.isnan(final_gain):
            logger.log("eta_search: Violated because gain is NaN")
        if np.isnan(final_constraint_val):
            logger.log("eta_search: Violated because KL is NaN")
        if final_gain < gain_before:
            logger.log("eta_search: Violated because gain not improving")
        if final_constraint_val >= epsilon:
            logger.log("eta_search: Violated because KL constraint violated")
        set_from_flat(all_params)
    else:
        set_from_flat(best_params)

    logger.log("eta optimization finished, final gain: " + str(final_gain))

    return eta


def visualize(env, policy_fn, *,
          timesteps_per_batch,  # what to train on
          epsilon, beta, cg_iters,
          gamma, lam,  # advantage estimation
          entcoeff=0.0,
          cg_damping=1e-2,
          vf_stepsize=3e-4,
          vf_iters =3,
          max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
          callback=None,
          TRPO=False,
          n_policy=1,
          policy_type=0,
          filepath='',
          session,
          retrace=False
          ):
    ob_space = env.observation_space
    ac_space = env.action_space
    pis = [policy_fn("pi_" + str(i), ob_space, ac_space) for i in range(n_policy)]
    tf.train.Saver().restore(session, filepath)
    done = False
    obs = env.reset()
    t = 0
    while not done:
        pi_id = t % len(pis)
        action = pis[pi_id].act(True, obs)[0]
        obs, reward, done, info = env.step(action)
        env.render()
        t += 1


def learn(env, policy_fn, *,
          timesteps_per_batch,  # what to train on
          epsilon, beta, cg_iters,
          gamma, lam,  # advantage estimation
          entcoeff=0.0,
          cg_damping=1e-2,
          vf_stepsize=3e-4,
          vf_iters =3,
          max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
          callback=None,
          TRPO=False,
          n_policy=1,
          policy_type=0,
          filepath='',
          session,
          retrace=False
          ):
    '''
    :param TRPO: True: TRPO, False: COPOS
    :param n_policy: Number of periodic policy parts
    :param policy_type: 0: Optimize 'n_policy' policies that are executed periodically. All the policies are updated.
                        1: The last 'n_policy' policies are executed periodically but only the last one is optimized.
                        2: The policy is spread over 'n_policy' time steps.
    '''
    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pis = [policy_fn("pi_" + str(i), ob_space, ac_space) for i in range(n_policy)]
    oldpis = [policy_fn("oldpi_" + str(i), ob_space, ac_space) for i in range(n_policy)]
    pi_vf = policy_fn("pi_vf", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    ob = U.get_placeholder_cached(name="ob")

    if policy_type == 0:
        print("Policy type: " + str(policy_type) + ". Optimize 'n_policy' policies that are executed periodically. All the policies are updated.")
    elif policy_type == 1:
        print("Policy type: " + str(policy_type) + ". The last 'n_policy' policies are executed periodically but only the last one is optimized.")
    elif policy_type == 2:
        print("Policy type: " + str(policy_type) + ". The policy is spread over 'n_policy' time steps.")
    else:
        print("Policy type: " + str(policy_type) + " is not supported.")

    # Compute variables for each policy separately
    old_entropy = []
    get_flat = []
    set_from_flat = []
    assign_old_eq_new = []
    copy_policy_back = []
    compute_losses = []
    compute_lossandgrad = []
    compute_fvp = []

    for i in range(n_policy):
        pi = pis[i]
        oldpi = oldpis[i]

        ac = pi.pdtype.sample_placeholder([None])

        kloldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        old_entropy.append(oldpi.pd.entropy())
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        entbonus = entcoeff * meanent

        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
        if retrace:
            surrgain = tf.reduce_mean(atarg) # atarg incorporates pnew / pold already
        else:
            surrgain = tf.reduce_mean(ratio * atarg)

        optimgain = surrgain + entbonus
        losses = [optimgain, meankl, entbonus, surrgain, meanent]
        loss_names = ["optimgain", "meankl", "entloss", "surrgain", "Entropy"]

        dist = meankl

        all_var_list = pi.get_trainable_variables()
        all_var_list = [v for v in all_var_list if v.name.split("/")[0].startswith("pi")]
        var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]

        #
        # fvp: Fisher Information Matrix / vector product based on Hessian of KL-divergence
        # fvp = F * v, where F = - E \partial_1 \partial_2 KL_div(p1 || p2)
        #
        get_flat.append(U.GetFlat(var_list))
        set_from_flat.append(U.SetFromFlat(var_list))
        klgrads = tf.gradients(dist, var_list)
        flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
        shapes = [var.get_shape().as_list() for var in var_list]
        start = 0
        tangents = []
        for shape in shapes:
            sz = U.intprod(shape)
            tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
            start += sz
        gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
        fvp = U.flatgrad(gvp, var_list)

        #
        # fvpll: Fisher Information Matrix / vector product based on exact FIM
        # fvpll = F * v, where F = E[\partial_1 \log p * \partial_2 \log p]
        #

        # Mean: (\partial \mu^T / \partial param1) * Precision * (\partial \mu / \partial param1)

        # Covariance: 0.5 * Trace[Precision * (\partial Cov / \partial param1) *
        #                         Precision * (\partial Cov / \partial param2)]

        if i > 0:
            # Only needed for policy_type == 1 for copying policy 'i' to policy 'i-1'
            copy_policy_back.append(U.function([], [], updates=[tf.assign(oldv, newv)
                for (oldv, newv) in zipsame(pis[i - 1].get_variables(), pi.get_variables())]))

        assign_old_eq_new.append(U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())]))
        compute_losses.append(U.function([ob, ac, atarg], losses))
        compute_lossandgrad.append(U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)]))
        compute_fvp.append(U.function([flat_tangent, ob, ac, atarg], fvp))

    # Value function is global to all policies
    vferr = tf.reduce_mean(tf.square(pi_vf.vpred - ret))
    all_var_list = pi_vf.get_trainable_variables()
    vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    vfadam = MpiAdam(vf_var_list)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    U.initialize()

    if policy_type == 1:
        # Initialize policies to identical values
        th_init = get_flat[0]()
        for i in range(n_policy):
            MPI.COMM_WORLD.Bcast(th_init, root=0)
            set_from_flat[i](th_init)
            vfadam.sync()
            print("Init param sum", th_init.sum(), flush=True)
    else:
        for i in range(n_policy):
            th_init = get_flat[i]()
            MPI.COMM_WORLD.Bcast(th_init, root=0)
            set_from_flat[i](th_init)
            vfadam.sync()
            print("Init param sum", th_init.sum(), flush=True)

    # Initialize eta, omega optimizer
    init_eta = 0.5
    init_omega = 2.0
    eta_omega_optimizer = EtaOmegaOptimizer(beta, epsilon, init_eta, init_omega)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = []
    for i in range(len(pis)):
        seg_gen.append(traj_segment_generator(pis, i + 1, pi_vf, env, timesteps_per_batch, stochastic=True))

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0])==1

    n_saves = 0
    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************"%iters_so_far)

        if max_timesteps > 0 and (timesteps_so_far >= (n_saves * max_timesteps // 5)):
            # Save policy
            saver = tf.train.Saver()
            saver.save(session, filepath + "_" + str(iters_so_far))
            n_saves += 1

        with timed("sampling"):
            if policy_type == 1 and iters_so_far < len(pis):
                all_seg = seg_gen[iters_so_far].__next__() # For four time steps use the four policies
            else:
                all_seg = seg_gen[-1].__next__()

        if policy_type == 1 and retrace:
            act_pi_ids = np.empty_like(all_seg["vpred"], dtype=int)
            act_pi_ids[:] = n_policy - 1 # Always update the last policy
            add_vtarg_and_adv_retrace(all_seg, gamma, lam, act_pi_ids)
        else:
            add_vtarg_and_adv(all_seg, gamma, lam)

        # Split the advantage functions etc. among the policies
        segs = split_traj_segment(pis, all_seg)

        # Update all policies
        for pi_id in range(n_policy):
            if policy_type == 1:
                # Update only last policy
                pi_id = n_policy - 1
                # Using all the samples
                seg = all_seg
            else:
                seg = segs[pi_id]

            pi = pis[pi_id]
            oldpi = oldpis[pi_id]

            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            vpredbefore = seg["vpred"] # predicted value function before update
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

            if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

            args = seg["ob"], seg["ac"], atarg
            fvpargs = [arr[::5] for arr in args]
            def fisher_vector_product(p):
                return allmean(compute_fvp[pi_id](p, *fvpargs)) + cg_damping * p

            assign_old_eq_new[pi_id]() # set old parameter values to new parameter values

            with timed("computegrad"):
                *lossbefore, g = compute_lossandgrad[pi_id](*args)
            lossbefore = allmean(np.array(lossbefore))
            g = allmean(g)
            if np.allclose(g, 0):
                logger.log("Got zero gradient. not updating")

                if policy_type == 1:
                    # Update only the last policy
                    break
            else:
                with timed("cg"):
                    stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
                assert np.isfinite(stepdir).all()

                if TRPO:
                    #
                    # TRPO specific code.
                    # Find correct step size using line search
                    #
                    shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                    lm = np.sqrt(shs / epsilon)
                    # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                    fullstep = stepdir / lm
                    expectedimprove = g.dot(fullstep)
                    surrbefore = lossbefore[0]
                    stepsize = 1.0
                    thbefore = get_flat[pi_id]()
                    for _ in range(10):
                        thnew = thbefore + fullstep * stepsize
                        set_from_flat[pi_id](thnew)
                        meanlosses = surr, kl, *_ = allmean(np.array(compute_losses[pi_id](*args)))
                        improve = surr - surrbefore
                        logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                        if not np.isfinite(meanlosses).all():
                            logger.log("Got non-finite value of losses -- bad!")
                        elif kl > epsilon * 1.5:
                            logger.log("violated KL constraint. shrinking step.")
                        elif improve < 0:
                            logger.log("surrogate didn't improve. shrinking step.")
                        else:
                            logger.log("Stepsize OK!")
                            break
                        stepsize *= .5
                    else:
                        logger.log("couldn't compute a good step")
                        set_from_flat[pi_id](thbefore)
                else:
                    #
                    # COPOS specific implementation.
                    #

                    copos_update_dir = stepdir

                    # Split direction into log-linear 'w_theta' and non-linear 'w_beta' parts
                    w_theta, w_beta = pi.split_w(copos_update_dir)

                    # q_beta(s,a) = \grad_beta \log \pi(a|s) * w_beta
                    #             = features_beta(s) * K^T * Prec * a
                    # q_beta = self.target.get_q_beta(features_beta, actions)

                    Waa, Wsa = pi.w2W(w_theta)
                    wa = pi.get_wa(ob, w_beta)

                    varphis = pi.get_varphis(ob)

                    # Optimize eta and omega
                    tmp_ob = np.zeros((1,) + env.observation_space.shape) # We assume that entropy does not depend on the NN
                    old_ent = old_entropy[pi_id].eval({oldpi.ob: tmp_ob})[0]
                    eta, omega = eta_omega_optimizer.optimize(w_theta, Waa, Wsa, wa, varphis, pi.get_kt(),
                                                              pi.get_prec_matrix(), pi.is_new_policy_valid, old_ent)
                    logger.log("Initial eta: " + str(eta) + " and omega: " + str(omega))

                    current_theta_beta = get_flat[pi_id]()
                    prev_theta, prev_beta = pi.all_to_theta_beta(current_theta_beta)

                    for i in range(2):
                        # Do a line search for both theta and beta parameters by adjusting only eta
                        eta = eta_search(w_theta, w_beta, eta, omega, allmean, compute_losses[pi_id],
                                         get_flat[pi_id], set_from_flat[pi_id], pi, epsilon, args)
                        logger.log("Updated eta, eta: " + str(eta) + " and omega: " + str(omega))

                        # Find proper omega for new eta. Use old policy parameters first.
                        set_from_flat[pi_id](pi.theta_beta_to_all(prev_theta, prev_beta))
                        eta, omega = \
                            eta_omega_optimizer.optimize(w_theta, Waa, Wsa, wa, varphis, pi.get_kt(),
                                                         pi.get_prec_matrix(), pi.is_new_policy_valid, old_ent, eta)
                        logger.log("Updated omega, eta: " + str(eta) + " and omega: " + str(omega))

                    # Use final policy
                    logger.log("Final eta: " + str(eta) + " and omega: " + str(omega))
                    cur_theta = (eta * prev_theta + w_theta.reshape(-1, )) / (eta + omega)
                    cur_beta = prev_beta + w_beta.reshape(-1, ) / eta
                    thnew = pi.theta_beta_to_all(cur_theta, cur_beta)
                    set_from_flat[pi_id](thnew)

                    meanlosses = surr, kl, *_ = allmean(np.array(compute_losses[pi_id](*args)))

                if nworkers > 1 and iters_so_far % 20 == 0:
                    paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam[pi_id].getflat().sum())) # list of tuples
                    assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

                for (lossname, lossval) in zip(loss_names, meanlosses):
                    logger.record_tabular(lossname, lossval)

                logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

                if policy_type == 1:
                    # Update only the last policy
                    break

        if policy_type == 1:
            # Copy policies 1, ..., i to 0, ..., i-1
            for j in range(n_policy - 1):
                copy_policy_back[j]()

        with timed("vf"):
            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((all_seg["ob"], all_seg["tdlamret"]),
                include_final_partial_batch=False, batch_size=64):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        lrlocal = (all_seg["ep_lens"], all_seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("AverageReturn", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank==0:
            logger.dump_tabular()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]