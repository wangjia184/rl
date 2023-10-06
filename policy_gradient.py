import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import pathlib
import platform

current_dir = pathlib.Path(__file__).parent.resolve()
os_type = platform.system() + '-' + platform.machine()


model = tf.keras.Sequential(
    [
        tf.keras.Input(type_spec=tf.RaggedTensorSpec(
            shape=[None, 7], dtype=tf.float32)),
        tf.keras.layers.Dense(
            256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.004)),
        # tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(
            512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.004)),
        # tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(
            1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.004)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(2, activation='softmax',
                              kernel_regularizer=tf.keras.regularizers.l2(0.004))
    ]
)

model.summary()

model.load_weights(str(current_dir) + "/initial_weights/")


class agent():
    def __init__(self):
        self.model = model
        # self.opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.000001)
        self.gamma = 0.95

    def act(self, state):
        prob = self.model(np.array([state]))
        # print(state)
        # print(prob)
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def actor_loss(self, prob, action, reward):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        # get the log probability of the specific action
        log_prob = dist.log_prob(action)
        loss = -log_prob*reward
        return loss

    def train(self, states, rewards, actions):
        sum_reward = 0
        discnt_rewards = []
        rewards.reverse()

        for r in rewards:
            sum_reward = r + self.gamma*sum_reward
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()

        # Initialize an accumulator for gradients
        total_grads = [tf.zeros_like(var)
                       for var in self.model.trainable_variables]

        for state, reward, action in zip(states, discnt_rewards, actions):
            with tf.GradientTape() as tape:
                p = self.model(np.array([state]), training=True)
                loss = self.actor_loss(p, action, reward)

            # Compute gradients and accumulate them
            grads = tape.gradient(loss, self.model.trainable_variables)
            total_grads = [acc + grad for acc, grad in zip(total_grads, grads)]

        # Update the model using the accumulated gradients
        self.opt.apply_gradients(
            zip(total_grads, self.model.trainable_variables))


# run the game in chrome and collect data for training
chromedriver_path = './{}/chromedriver'.format(os_type)
print(chromedriver_path)
service = Service(executable_path=chromedriver_path)
driver = webdriver.Chrome(service=service)

driver.get("file://" + str(current_dir /
           "JS-Flappy-Bird-master" / "index.html"))


my_agent = agent()

for episode in range(100000):
    driver.execute_script(" start();")

    rewards = []
    states = []
    actions = []

    max_frames = 5
    total_rewards = 0

    while True:
        state_dic = driver.execute_script("return step(0);")
        if state_dic['to_start'] == 143:
            break

    running = state_dic['running']
    state = [state_dic['to_gnd'] / 100.0, state_dic['to_roof'] / 10.0, state_dic['to_floor'] / 10.0, state_dic['to_start'] /
             100.0, state_dic['to_end'] / 100.0, state_dic['speed'] / 10.0, state_dic['to_next_roof'] / 100.0]
    while running:
        action = my_agent.act(state)
        # print(state[3])
        states.append(state)
        actions.append(action)

        reward = 0
        for step in range(max_frames):
            if running:
                flap = action > 0 and step == 0
                state_dic = driver.execute_script(
                    "return step({});".format("1" if flap else "0"))
                running = state_dic['running']
                reward += state_dic['reward']
                if state_dic['to_start'] == 143:
                    break

        rewards.append(reward)
        total_rewards += reward
        state = [state_dic['to_gnd'] / 100.0, state_dic['to_roof'] / 10.0, state_dic['to_floor'] / 10.0, state_dic['to_start'] /
                 100.0, state_dic['to_end'] / 100.0, state_dic['speed'] / 10.0, state_dic['to_next_roof'] / 100.0]

    my_agent.train(states, rewards, actions)
    print("{} Total rewards = {}".format(episode, total_rewards))
    print("--------------------------")

model.save_weights(str(current_dir) + "/checkpoint/", overwrite=True)

driver.quit()
