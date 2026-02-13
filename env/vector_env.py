
import multiprocessing as mp
import numpy as np

def worker(remote, parent_remote, env_fn_wrapper):
    """
    Worker process to run an environment instance.
    """
    parent_remote.close()
    env = env_fn_wrapper.x() # Unwrap and create env
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                next_state, reward, done, info = env.step(data)
                if done:
                    # Auto-reset if done (standard practice in VecEnv)
                    # Note: You might want to store the terminal state in info if needed
                    next_state = env.reset()
                remote.send((next_state, reward, done, info))
            elif cmd == 'reset':
                state = env.reset()
                remote.send(state)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'call_method':
                 method_name, args, kwargs = data
                 result = getattr(env, method_name)(*args, **kwargs)
                 remote.send(result)
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close() if hasattr(env, 'close') else None

class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to pickle
    the entire env object which might not be picklable).
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class SubprocVecEnv:
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended for PC with multi-core CPU to avoid serial bottleneck.
    """
    def __init__(self, env_fns):
        """
        env_fns: List of callables that create environments
        """
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        
        # Fork is not available on Windows, spawn is default.
        # We need to use 'spawn' or rely on default context which is spawn on Windows.
        ctx = mp.get_context('spawn')
        
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.ps = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang?
            # Actually daemon=True means they are killed if main process exits.
            p = ctx.Process(target=worker, args=args, daemon=True) 
            p.start()
            self.ps.append(p)
            work_remote.close()

        # Get properties from first env (assuming all are same)
        self.remotes[0].send(('get_attr', 'state_size'))
        self.state_size = self.remotes[0].recv()
        
        self.remotes[0].send(('get_attr', 'action_size'))
        self.action_size = self.remotes[0].recv()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
        
    def getStateSize(self):
        return self.state_size
    
    def getActionSize(self):
        return self.action_size
    
    def getValidActions(self, state=None):
        # Assuming valid actions are same for all envs or returning a batch of masks
        # Since original code returns one mask, we should return a batch of masks here.
        # But wait, the original `getValidActions` is called inside `agent.play`.
        # If agent is vectorized, it should handle batch of masks.
        
        # Let's just ask the first env for valid actions shape/mask and replicate it 
        # OR we can ask all envs if masks are dynamic.
        # In this specific problem, masks seem to be static ones(size).
        
        # We'll implement a method to get batch of masks
        # For simplicity, if mask is static ones, we construct it here.
        mask = np.ones((self.num_envs, self.action_size), dtype=np.float32)
        return mask

    def get_attr(self, attr_name):
        """
        Get attribute from each environment.
        """
        for remote in self.remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in self.remotes]

    def call_method_batch(self, method_name, args_list):
        """
        Call a method on each environment with different arguments.
        args_list: List of args tuples, one for each env.
        """
        assert len(args_list) == self.num_envs
        for remote, args in zip(self.remotes, args_list):
            remote.send(('call_method', (method_name, args, {})))
        return [remote.recv() for remote in self.remotes]

    def __del__(self):
        self.close()

class DummyVecEnv:
    """
    VecEnv that runs multiple environments sequentially in the same process.
    Useful for evaluation or debugging to avoid multiprocessing overhead.
    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(env_fns)
        
        # Get properties
        # Assuming all envs are same
        self.state_size = self.envs[0].state_size
        self.action_size = self.envs[0].action_size
        
    def step_async(self, actions):
        self.actions = actions
        
    def step_wait(self):
        results = []
        for env, action in zip(self.envs, self.actions):
            next_state, reward, done, info = env.step(action)
            if done:
                # Auto-reset if done
                next_state = env.reset()
            results.append((next_state, reward, done, info))
            
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos
    
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
    
    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.stack(obs)
        
    def close(self):
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()

    def getStateSize(self):
        return self.state_size
    
    def getActionSize(self):
        return self.action_size
    
    def getValidActions(self, state=None):
        mask = np.ones((self.num_envs, self.action_size), dtype=np.float32)
        return mask
        
    def get_attr(self, attr_name):
        return [getattr(env, attr_name) for env in self.envs]
    
    def call_method_batch(self, method_name, args_list):
        """
        Call a method on each environment with different arguments.
        args_list: List of args tuples, one for each env.
        """
        assert len(args_list) == self.num_envs
        results = []
        for env, args in zip(self.envs, args_list):
            results.append(getattr(env, method_name)(*args))
        return results
