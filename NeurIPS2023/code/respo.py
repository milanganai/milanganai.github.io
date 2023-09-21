import numpy as np
import torch
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
import safety_gym
import gym
import time
import  core_respo
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from torch.nn.functional import softplus
torch.autograd.set_detect_anomaly(True)


class RESPOBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97):
        self.obs_buf = np.zeros(core_respo.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core_respo.combined_shape(size, act_dim), dtype=np.float32)
        
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.cadv_buf = np.zeros(size, dtype=np.float32)

        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.crew_buf = np.zeros(size, dtype=np.float32)
        self.prew_buf = np.zeros(size, dtype=np.float32)

        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.cret_buf = np.zeros(size, dtype=np.float32)
        self.pret_buf = np.zeros(size, dtype=np.float32)

        self.val_buf = np.zeros(size, dtype=np.float32)
        self.cval_buf = np.zeros(size, dtype=np.float32)
        self.pval_buf = np.zeros(size, dtype=np.float32)
        
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        self.ac = None

    def store(self, obs, act, rew, crew, val,cval, logp):

        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.crew_buf[self.ptr] = crew
        self.prew_buf[self.ptr] = float(crew>0.0)

        self.val_buf[self.ptr] = val
        self.cval_buf[self.ptr] = cval

        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0, last_cval=0):

        path_slice = slice(self.path_start_idx, self.ptr)

        mu = 0.99
        self.pval_buf[path_slice] = self.ac.vp(torch.as_tensor(self.obs_buf[path_slice], dtype=torch.float32)).detach().numpy()

        rews = np.append(self.rew_buf[path_slice], last_val)
        crews = np.append(self.crew_buf[path_slice], last_cval)
        prews = np.append(self.prew_buf[path_slice], float(last_cval>0.0))

        vals = np.append(self.val_buf[path_slice], last_val)
        cvals = np.append(self.cval_buf[path_slice], last_cval)
        pvals = np.append(self.pval_buf[path_slice], float(last_cval>0.0))
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        cdeltas = crews[:-1] + self.gamma * cvals[1:] - cvals[:-1]

        self.adv_buf[path_slice] = core_respo.discount_cumsum(deltas, self.gamma * self.lam)
        self.cadv_buf[path_slice] = core_respo.discount_cumsum(cdeltas, self.gamma * self.lam)

        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core_respo.discount_cumsum(rews, self.gamma)[:-1]
        self.cret_buf[path_slice] = core_respo.discount_cumsum(crews, self.gamma)[:-1]

        pret_buf = prews.copy()
        for i in range(pret_buf.shape[0]-2, -1, -1):
            pret_buf[i] = max(pret_buf[i], mu * pret_buf[i+1])
        pret_buf = pret_buf[:-1]
        self.pret_buf[path_slice] = pret_buf
        
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        cadv_mean, cadv_std = mpi_statistics_scalar(self.cadv_buf)

        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        self.cadv_buf = (self.cadv_buf - cadv_mean) #/ adv_std

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, cret=self.cret_buf, pval=self.pval_buf, pret=self.pret_buf,
                    prew=self.prew_buf, adv=self.adv_buf, cadv=self.cadv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



def respo(env_fn, actor_critic=core_respo.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=30000, epochs=300, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=1):

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core_respo.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = RESPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    buf.ac = ac

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, cadv,  logp_old = data['obs'], data['act'], data['adv'], data['cadv'] ,data['logp']
        punsafe = data['pval']
        psafe = 1.0 - data['pval']
        cur_cost = data['cur_cost']
        penalty_param = data['cur_penalty']
        cost_limit = 0
        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)

        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv * psafe
        loss_rpi = (torch.min(ratio * adv * psafe, clip_adv)).mean()

        p = softplus(penalty_param)
        penalty_item = p.item()
      
        loss_cpi = ratio*cadv * (penalty_item * psafe + punsafe)
        loss_cpi = loss_cpi.mean()
        
        pi_objective = loss_rpi - loss_cpi
        pi_objective = pi_objective/(1+penalty_item)
        loss_pi = -pi_objective


        cost_deviation = (cur_cost - cost_limit) * psafe.mean()


        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, cost_deviation, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret, cret, pret = data['obs'], data['ret'], data['cret'], data['pret']
        return ((ac.v(obs) - ret)**2).mean(),((ac.vc(obs) - cret)**2).mean(), ((ac.vp(obs) - pret)**2).mean()


    # Set up optimizers for policy and value function
    pi_lr = 3e-4
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    penalty_param = torch.tensor(.25,requires_grad=True).float()
    penalty = softplus(penalty_param)
    

    penalty_lr = 5e-5
    penalty_optimizer = Adam([penalty_param], lr=penalty_lr)
    vf_lr = 1e-3
    pvf_lr = 1e-4
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    cvf_optimizer = Adam(ac.vc.parameters(),lr=vf_lr)
    pvf_optimizer = Adam(ac.vp.parameters(),lr=pvf_lr)

    schedulers = [
        lr_scheduler.LinearLR(pi_optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs),
        lr_scheduler.LinearLR(penalty_optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs),
        lr_scheduler.LinearLR(vf_optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs),
        lr_scheduler.LinearLR(cvf_optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs),
        lr_scheduler.LinearLR(pvf_optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)
    ]
    # Set up model saving
    logger.setup_pytorch_saver(ac)

  

    def update():
        cur_cost = logger.get_stats('EpCost')[0]
        data = buf.get()
        data['cur_cost'] = cur_cost
        data['cur_penalty'] = penalty_param
        pi_l_old, cost_dev, pi_info_old = compute_loss_pi(data)
        loss_penalty = -penalty_param*cost_dev

        
        penalty_optimizer.zero_grad()
        loss_penalty.backward()
        mpi_avg_grads(penalty_param)
        penalty_optimizer.step()


        data['cur_penalty'] = penalty_param

        pi_l_old = pi_l_old.item()
        v_l_old, cv_l_old, pv_l_old = compute_loss_v(data)
        v_l_old, cv_l_old, pv_l_old = v_l_old.item(), cv_l_old.item(), pv_l_old.item() 


        # Train policy with multiple steps of gradient descent
        train_pi_iters=80
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, _,pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.2 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break

            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        train_v_iters=80
        for i in range(train_v_iters):

            loss_v, loss_vc, loss_vp = compute_loss_v(data)
            vf_optimizer.zero_grad()
            loss_v.backward()
            mpi_avg_grads(ac.v)   # average grads across MPI processes
            vf_optimizer.step()

            cvf_optimizer.zero_grad()
            loss_vc.backward()
            mpi_avg_grads(ac.vc)
            cvf_optimizer.step()

            pvf_optimizer.zero_grad()
            loss_vp.backward()
            mpi_avg_grads(ac.vp)
            pvf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old, PSafe=(1.0-data['pval'].mean()), ViolAvg=data['prew'].mean(),
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))
        


    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret,ep_cret, ep_len = env.reset(), 0, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, vc, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, info = env.step(a)
            c = info['cost']
            ep_ret += r
            ep_cret += c
            ep_len += 1

            # save and log
            buf.store(o, a, r, c, v,vc, logp)
            logger.store(VVals=v)
            logger.store(CVVals=vc)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v,vc, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                    vc = 0
                buf.finish_path(last_val=v,last_cval=vc)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cret)
                try:
                    o, ep_ret, ep_cret, ep_len = env.reset(), 0, 0, 0
                except:
                    try:
                        o, ep_ret, ep_cret, ep_len = env.reset(), 0, 0, 0
                    except:
                        try:
                            o, ep_ret, ep_cret, ep_len = env.reset(), 0, 0, 0
                        except:
                            env = env_fn()
                            o, ep_ret, ep_cret, ep_len = env.reset(), 0, 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()

        # update schedulers
        for s in schedulers:
            s.step()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpCost',with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('Lagrange', penalty_param.item())
        logger.log_tabular('PSafe', average_only=True)
        logger.log_tabular('ViolAvg', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Safexp-CarGoal1-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--exp_name', type=str, default='respo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    num_steps = 9e6
    steps_per_epoch = 30000
    epochs = int(num_steps / steps_per_epoch)
    respo(lambda : gym.make(args.env), actor_critic=core_respo.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=steps_per_epoch, epochs=epochs,
        logger_kwargs=logger_kwargs)
