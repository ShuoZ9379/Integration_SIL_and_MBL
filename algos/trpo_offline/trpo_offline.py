#Question Lines 351(could be changed already)
from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time, os, pickle, copy, sys
from baselines.common import colorize
from mpi4py import MPI
from collections import deque
from baselines.common import set_global_seeds
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.common.input import observation_placeholder
from baselines.common.policies import build_policy
from contextlib import contextmanager
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# MBL
from mbl.mbl import MBL, MBLCEM, MBLMPPI
from mbl.exp_util import eval_policy, Policy
from mbl.util.util import load_extracted_val_data as load_val_data
from mbl.util.util import to_onehot
from mbl.model_config import get_make_mlp_model
#from plot import plot 
#from visdom import Visdom
from multiprocessing.dummy import Pool
import multiprocessing as mp

class DummyPolicy(object):
    def __init__(self, fn):
        self.fn = fn

    def step(self, ob, stochastic=True):
        ac, v =  self.fn(ob, stochastic)
        return ac, v, None, None

# ------------

def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
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

    while True:
        prevac = ac
        ac, vpred, _, _ = pi.step(ob, stochastic=stochastic)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            _, vpred, _, _ = pi.step(ob, stochastic=stochastic)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
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

def learn(*,
        network,
        env, eval_env, make_eval_env, env_id,
        total_timesteps,
        timesteps_per_batch=1024, # what to train on
        max_kl=0.001,
        cg_iters=10,
        gamma=0.99,
        lam=1.0, # advantage estimation
        seed=None,
        ent_coef=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        load_path=None,

        # MBL
        # For train mbl
        mbl_train_freq=5,

        # For eval
        num_eval_episodes=5,
        eval_freq=5,
        vis_eval=False,
#        eval_targs=('mbmf',),
        eval_targs=('mf',),
        quant=2,

        # For mbl.step
        #num_samples=(1500,),
        num_samples=(1,),
        horizon=(2,),
        #horizon=(2,1),
        #num_elites=(10,),
        num_elites=(1,),
        mbl_lamb=(1.0,),
        mbl_gamma=0.99,
        #mbl_sh=1, # Number of step for stochastic sampling
        mbl_sh=10000,
        #vf_lookahead=-1,
        #use_max_vf=False,
        reset_per_step=(0,),

        # For get_model
        num_fc=2,
        num_fwd_hidden=500,
        use_layer_norm=False,        

        # For MBL
        num_warm_start=int(1e4),            
        init_epochs=10, 
        update_epochs=5, 
        batch_size=512, 
        update_with_validation=False, 
        use_mean_elites=1,
        use_ent_adjust=0,
        adj_std_scale=0.5,

        # For data loading
        validation_set_path=None, 

        # For data collect
        collect_val_data=False,

        # For traj collect
        traj_collect='mf',      
        
        # For profile
        measure_time=True,
        eval_val_err=False,
        measure_rew=True,
               
        **network_kwargs
        ):
    '''
    learn a policy function with TRPO algorithm

    Parameters:
    ----------

    network                 neural network to learn. Can be either string ('mlp', 'cnn', 'lstm', 'lnlstm' for basic types)
                            or function that takes input placeholder and returns tuple (output, None) for feedforward nets
                            or (output, (state_placeholder, state_output, mask_placeholder)) for recurrent nets

    env                     environment (one of the gym environments or wrapped via baselines.common.vec_env.VecEnv-type class

    timesteps_per_batch     timesteps per gradient estimation batch

    max_kl                  max KL divergence between old policy and new policy ( KL(pi_old || pi) )

    ent_coef                coefficient of policy entropy term in the optimization objective

    cg_iters                number of iterations of conjugate gradient algorithm

    cg_damping              conjugate gradient damping

    vf_stepsize             learning rate for adam optimizer used to optimie value function loss

    vf_iters                number of iterations of value function optimization iterations per each policy optimization step

    total_timesteps           max number of timesteps

    max_episodes            max number of episodes

    max_iters               maximum number of policy optimization iterations

    callback                function to be called with (locals(), globals()) each policy optimization step

    load_path               str, path to load the model from (default: None, i.e. no model is loaded)

    **network_kwargs        keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network

    Returns:
    -------

    learnt model

    '''
    if not isinstance(num_samples, tuple): num_samples = (num_samples,)
    if not isinstance(horizon, tuple): horizon = (horizon,)
    if not isinstance(num_elites, tuple): num_elites = (num_elites,)
    if not isinstance(mbl_lamb, tuple): mbl_lamb = (mbl_lamb,)
    if not isinstance(reset_per_step, tuple): reset_per_step = (reset_per_step,)
    if validation_set_path is None: 
        if collect_val_data: validation_set_path = os.path.join(logger.get_dir(), 'val.pkl')
        else: validation_set_path = os.path.join('dataset', '{}-val.pkl'.format(env_id))
    if eval_val_err:
        eval_val_err_path = os.path.join('dataset', '{}-combine-val.pkl'.format(env_id))
    logger.log(locals())
    logger.log('MBL_SH', mbl_sh)
    logger.log('Traj_collect', traj_collect)
    
    if MPI is not None:
        nworkers = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        nworkers = 1
        rank = 0  
    cpus_per_worker = 1
    U.get_session(config=tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=cpus_per_worker,
            intra_op_parallelism_threads=cpus_per_worker
    ))

    set_global_seeds(seed)
    
    policy = build_policy(env, network, value_network='copy', **network_kwargs)

    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    ob = observation_placeholder(ob_space)
    with tf.variable_scope("pi"):
        pi = policy(observ_placeholder=ob)
    with tf.variable_scope("oldpi"):
        oldpi = policy(observ_placeholder=ob)

    # MBL
    # ---------------------------------------
    #viz = Visdom(env=env_id) 
    win = None
    eval_targs = list(eval_targs)
    logger.log(eval_targs)

    make_model = get_make_mlp_model(num_fc=num_fc, num_fwd_hidden=num_fwd_hidden, layer_norm=use_layer_norm)
    mbl = MBL(env=eval_env, env_id=env_id, make_model=make_model,
            num_warm_start=num_warm_start,            
            init_epochs=init_epochs, 
            update_epochs=update_epochs, 
            batch_size=batch_size, 
            **network_kwargs)

    val_dataset = {'ob': None, 'ac': None, 'ob_next': None}
    if update_with_validation:
        logger.log('Update with validation')
        val_dataset = load_val_data(validation_set_path)
    if eval_val_err:
        logger.log('Log val error')
        eval_val_dataset = load_val_data(eval_val_err_path)       
    if collect_val_data:
        logger.log('Collect validation data')
        val_dataset_collect = [] 

    def _mf_pi(ob, t=None):
        stochastic = True
        ac, vpred, _, _ = pi.step(ob, stochastic=stochastic)
        return ac, vpred   
    def _mf_det_pi(ob, t=None):
        #ac, vpred, _, _ = pi.step(ob, stochastic=False)
        ac, vpred = pi._evaluate([pi.pd.mode(), pi.vf], ob)        
        return ac, vpred
    def _mf_ent_pi(ob, t=None):
        mean, std, vpred = pi._evaluate([pi.pd.mode(), pi.pd.std, pi.vf], ob)
        ac = np.random.normal(mean, std * adj_std_scale, size=mean.shape)
        return ac, vpred
################### use_ent_adjust======> adj_std_scale????????pi action sample
    def _mbmf_inner_pi(ob, t=0):
        if use_ent_adjust:
            return _mf_ent_pi(ob)
        else:
            #return _mf_pi(ob)
            if t < mbl_sh: return _mf_pi(ob)        
            else: return _mf_det_pi(ob)

    # ---------------------------------------   
 
    # Run multiple configuration once
    all_eval_descs = []
    def make_mbmf_pi(n, h, e, l):
        def _mbmf_pi(ob):                        
            ac, rew = mbl.step(ob=ob, pi=_mbmf_inner_pi, horizon=h, num_samples=n, num_elites=e, gamma=mbl_gamma, lamb=l, use_mean_elites=use_mean_elites) 
            return ac[None], rew
        return Policy(step=_mbmf_pi, reset=None)

    for n in num_samples:
        for h in horizon:
            for l in mbl_lamb:
                for e in num_elites:                     
                    if 'mbmf' in eval_targs: all_eval_descs.append(('MeanRew', 'MBL_TRPO', make_mbmf_pi(n, h, e, l)))
                    #if 'mbmf' in eval_targs: all_eval_descs.append(('MeanRew-n-{}-h-{}-e-{}-l-{}-sh-{}-me-{}'.format(n, h, e, l, mbl_sh, use_mean_elites), 'MBL_TRPO-n-{}-h-{}-e-{}-l-{}-sh-{}-me-{}'.format(n, h, e, l, mbl_sh, use_mean_elites), make_mbmf_pi(n, h, e, l)))                   
    if 'mf' in eval_targs: all_eval_descs.append(('MeanRew', 'TRPO', Policy(step=_mf_pi, reset=None)))
   
    logger.log('List of evaluation targets')
    for it in all_eval_descs:
        logger.log(it[0])    

    pool = Pool(mp.cpu_count())
    warm_start_done = False
    # ----------------------------------------

    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = ent_coef * meanent

    vferr = tf.reduce_mean(tf.square(pi.vf - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = get_trainable_variables("pi")
    # var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    # vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    var_list = get_pi_trainable_variables("pi")
    vf_var_list = get_vf_trainable_variables("pi")

    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
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

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(get_variables("oldpi"), get_variables("pi"))])

    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
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
    if load_path is not None:
        pi.load(load_path)

    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)
    # Prepare for rollouts
    # ----------------------------------------
    if traj_collect == 'mf':
        seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards

    if sum([max_iters>0, total_timesteps>0, max_episodes>0])==0:
        # noththing to be done
        return pi

    assert sum([max_iters>0, total_timesteps>0, max_episodes>0]) < 2, \
        'out of max_iters, total_timesteps, and max_episodes only one should be specified'
    
    
    while True:
        if callback: callback(locals(), globals())
        if total_timesteps and timesteps_so_far >= total_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()
            if traj_collect == 'mf-random' or traj_collect == 'mf-mb':
                seg_mbl = seg_gen_mbl.__next__()
            else:
                seg_mbl = seg
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        
        # Val data collection
        if collect_val_data:
            for ob_, ac_, ob_next_ in zip(ob[:-1, 0, ...], ac[:-1, ...], ob[1:, 0, ...]):            
                val_dataset_collect.append((copy.copy(ob_), copy.copy(ac_), copy.copy(ob_next_)))
        # -----------------------------
        # MBL update
        else:
            ob_mbl, ac_mbl = seg_mbl["ob"], seg_mbl["ac"]
     
            mbl.add_data_batch(ob_mbl[:-1, 0, ...], ac_mbl[:-1, ...], ob_mbl[1:, 0, ...])
            mbl.update_forward_dynamic(require_update=iters_so_far % mbl_train_freq == 0, 
                    ob_val=val_dataset['ob'], ac_val=val_dataset['ac'], ob_next_val=val_dataset['ob_next'])            
        # -----------------------------
        
        if traj_collect == 'mf':
        #if traj_collect == 'mf' or traj_collect == 'mf-random' or traj_collect == 'mf-mb':
            vpredbefore = seg["vpred"] # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

            if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
            if hasattr(pi, "rms"):  pi.rms.update(ob) # update running mean/std for policy
            
            args = seg["ob"], seg["ac"], atarg
            fvpargs = [arr[::5] for arr in args]
            def fisher_vector_product(p):
                return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

            assign_old_eq_new() # set old parameter values to new parameter values
            with timed("computegrad"):
                *lossbefore, g = compute_lossandgrad(*args)
            lossbefore = allmean(np.array(lossbefore))
            g = allmean(g)
            if np.allclose(g, 0):
                logger.log("Got zero gradient. not updating")
            else:
                with timed("cg"):
                    stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
                assert np.isfinite(stepdir).all()
                shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / max_kl)
                # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                fullstep = stepdir / lm
                expectedimprove = g.dot(fullstep)
                surrbefore = lossbefore[0]
                stepsize = 1.0
                thbefore = get_flat()
                for _ in range(10):
                    thnew = thbefore + fullstep * stepsize
                    set_from_flat(thnew)
                    meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                    improve = surr - surrbefore
                    logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                    if not np.isfinite(meanlosses).all():
                        logger.log("Got non-finite value of losses -- bad!")
                    elif kl > max_kl * 1.5:
                        logger.log("violated KL constraint. shrinking step.")
                    elif improve < 0:
                        logger.log("surrogate didn't improve. shrinking step.")
                    else:
                        logger.log("Stepsize OK!")
                        break
                    stepsize *= .5
                else:
                    logger.log("couldn't compute a good step")
                    set_from_flat(thbefore)
                if nworkers > 1 and iters_so_far % 20 == 0:
                    paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                    assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

            for (lossname, lossval) in zip(loss_names, meanlosses):
                logger.record_tabular(lossname, lossval)

            with timed("vf"):

                for _ in range(vf_iters):
                    for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                    include_final_partial_batch=False, batch_size=64):
                        g = allmean(compute_vflossandgrad(mbob, mbret))
                        vfadam.update(g, vf_stepsize)

            logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        if MPI is not None:
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        else:
            listoflrpairs = [lrlocal]
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        
        if rank==0:
            # MBL evaluation
            if not collect_val_data:
                #set_global_seeds(seed)
                default_sess = tf.get_default_session()
                def multithread_eval_policy(env_, pi_, num_episodes_, vis_eval_,seed):
                    with default_sess.as_default():
                        if hasattr(env, 'ob_rms') and hasattr(env_, 'ob_rms'):
                            env_.ob_rms = env.ob_rms 
                        res = eval_policy(env_, pi_, num_episodes_, vis_eval_, seed, measure_time, measure_rew) 
                        
                        try:
                            env_.close()
                        except:
                            pass
                    return res

                if mbl.is_warm_start_done() and iters_so_far % eval_freq == 0:
                    warm_start_done = mbl.is_warm_start_done()
                    if num_eval_episodes > 0 :
                        targs_names = {}
                        with timed('eval'):
                            num_descs = len(all_eval_descs)
                            list_field_names = [e[0] for e in all_eval_descs]
                            list_legend_names = [e[1] for e in all_eval_descs]
                            list_pis = [e[2] for e in all_eval_descs]                    
                            list_eval_envs = [make_eval_env() for _ in range(num_descs)]
                            list_seed= [seed for _ in range(num_descs)]
                            list_num_eval_episodes = [num_eval_episodes for _ in range(num_descs)]
                            print(list_field_names)
                            print(list_legend_names)
                            
                            list_vis_eval = [vis_eval for _ in range(num_descs)]

                            for i in range(num_descs):
                                field_name, legend_name=list_field_names[i], list_legend_names[i],
                                
                                res= multithread_eval_policy(list_eval_envs[i], list_pis[i], list_num_eval_episodes[i], list_vis_eval[i], seed)
                            #eval_results = pool.starmap(multithread_eval_policy, zip(list_eval_envs, list_pis, list_num_eval_episodes, list_vis_eval,list_seed))
                            
                            #for field_name, legend_name, res in zip(list_field_names, list_legend_names, eval_results):
                                perf, elapsed_time, eval_rew = res
                                logger.record_tabular(field_name, perf)                    
                                if measure_time: logger.record_tabular('Time-%s' % (field_name), elapsed_time)
                                if measure_rew: logger.record_tabular('SimRew-%s' % (field_name), eval_rew)
                                targs_names[field_name] = legend_name
    
                    if eval_val_err:
                        fwd_dynamics_err = mbl.eval_forward_dynamic(obs=eval_val_dataset['ob'], acs=eval_val_dataset['ac'], obs_next=eval_val_dataset['ob_next'])        
                        logger.record_tabular('FwdValError', fwd_dynamics_err)

                    logger.dump_tabular()
                    #print(logger.get_dir())
                    #print(targs_names)
                    #if num_eval_episodes > 0:
#                        win = plot(viz, win, logger.get_dir(), targs_names=targs_names, quant=quant, opt='best')                
            # -----------
        #logger.dump_tabular()
        yield pi   

    if collect_val_data:
        with open(validation_set_path, 'wb') as f:
            pickle.dump(val_dataset_collect, f)
        logger.log('Save {} validation data'.format(len(val_dataset_collect)))

    #return pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

def get_vf_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'vf' in v.name[len(scope):].split('/')]

def get_pi_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]

