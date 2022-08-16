def run_scratch():
    env_test()

def env_test():
    import os
    import numpy as np
    import gym


    from fetch_envs import FetchReachEnv

    env = FetchReachEnv()
    import pdb; pdb.set_trace()




if __name__ == "__main__":
    run_scratch()
