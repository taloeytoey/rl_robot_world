import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

training_record = pd.read_csv('/home/taloeytoey/rl_robot/rl_robot_world/log/episode_reward_log2.txt', sep='\t')
# training_record.head()
fig = plt.figure()
plt.plot(training_record['EPISODE'], training_record['TOTAL_REWARD'], '.-', c = 'blue')
plt.xlabel('Episode')
plt.ylabel('Learning Performance')
fig.savefig('/home/taloeytoey/rl_robot/rl_robot_world/log/training_record.png', format='png', dpi=300, bbox_inches='tight')