import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import pathlib
import platform
import random

current_dir = pathlib.Path(__file__).parent.resolve()
os_type = platform.system() + '-' + platform.machine()


def build_actor_model():
    input = tf.keras.Input(shape=(None, 8), name="states")
    x = tf.keras.layers.Dense( 512, activation='relu')(input)

    
    for _ in range(9):
        #residual = x    
        #x = tf.keras.layers.BatchNormalization(trainable=trainable)(x)
        x = tf.keras.layers.Dense( 512, activation='relu')(x)
        #x = tf.keras.layers.Dense( 256, trainable=trainable)(x)
        #x = tf.keras.layers.Add()([residual, x])
        #x = tf.keras.layers.ReLU()(x)
        

    output = tf.keras.layers.Dense(2, activation='softmax')(x)
    return tf.keras.Model(input, output, name="actor")


actor_model = build_actor_model()
actor_model.summary()


critic_model = tf.keras.Sequential(
    [
        tf.keras.Input(type_spec=tf.RaggedTensorSpec(shape=[None, 8], dtype=tf.float32)),
        tf.keras.layers.Dense(
            256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.004)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(
            512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.004)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense( 1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.004)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.004))
    ]
)
critic_model.summary()


try:
    actor_model.load_weights(str(current_dir) + "/checkpoints/a2c_discrete_actor/latest")
    critic_model.load_weights(str(current_dir) + "/checkpoints/a2c_discrete_critic/latest")
except Exception:
    print("Failed to load actor_model or critic_model")



class agent():
    def __init__(self):
        self.actor = actor_model
        self.critic = critic_model
        self.actor_opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.00001) #  tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.critic_opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
        self.gamma = 0.95

    def act(self, state):
        prob = self.actor(np.array([state]))
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def actor_loss(self, prob, action, td_error):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action) # get the log probability of the specific action
        loss = -log_prob*td_error
        return loss
    
    



    def train(self, old_state, new_state, reward):

        new_value = self.critic(np.array([new_state]), training=False)
        with tf.GradientTape() as critic_tape:
            old_value = self.critic(np.array([old_state]), training=True)
            td_error = reward + tf.math.multiply(self.gamma, new_value) - old_value # TD error = Q - V_{t} = V_{t+1} x Gamma + reward - V_{t+1}
            critic_loss = tf.math.square(td_error) # MSE

        #print(td_error)
        with tf.GradientTape() as actor_tape:
            prob = self.actor(np.array([state]), training=True)
            actor_loss = self.actor_loss(prob, action, td_error)

        #print(critic_loss)
        # Compute gradients
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)  # MSE

        # Update models
        self.actor_opt.apply_gradients( zip(actor_grads, self.actor.trainable_variables))
        self.critic_opt.apply_gradients( zip(critic_grads, self.critic.trainable_variables))


# run the game in chrome and collect data for training
chromedriver_path = './{}/chromedriver'.format(os_type)
print("Starting chromedriver at :" + chromedriver_path)
service = Service(executable_path=chromedriver_path)
driver = webdriver.Chrome(service=service)

driver.get("file://" + str(current_dir / "docs" / "train.html"))

my_agent = agent()

for episode in range(2000):
    driver.execute_script(" start();")

    rewards = []
    states = []
    actions = []

    max_frames = 1
    total_rewards = 0

    state_dic = driver.execute_script("return step(0);")

    running = state_dic['running']
    state = [state_dic['to_gnd'], state_dic['to_roof'], state_dic['to_floor'], state_dic['to_start'], state_dic['next_frame_to_roof'], state_dic['next_frame_to_floor'], state_dic['speed'], state_dic['to_next_roof']]
    while running:
        action = my_agent.act(state)


        flap = action > 0
        state_dic = driver.execute_script("return step({});".format("1" if flap else "0"))
        running = state_dic['running']
        reward = state_dic['reward']
                

        rewards.append(reward)
        total_rewards += reward
        new_state = [state_dic['to_gnd'], state_dic['to_roof'], state_dic['to_floor'], state_dic['to_start'], state_dic['next_frame_to_roof'], state_dic['next_frame_to_floor'], state_dic['speed'], state_dic['to_next_roof']]
        
       
        my_agent.train( state, new_state, reward)

        state = new_state

    
    print("{} Total rewards = {}".format(episode, total_rewards))
    print("--------------------------")

actor_model.save_weights(str(current_dir) + "/checkpoints/a2c_discrete_actor/latest", overwrite=True)
critic_model.save_weights(str(current_dir) + "/checkpoints/a2c_discrete_critic/latest", overwrite=True)
driver.quit()
