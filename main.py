from training import env
from training import make_instances

# make_instances.make_n_instances(1)

input("Press Enter when ready...")


env = env.TMNF_ENV()

while True:
    env.step(env.action_space.sample())
