
from multiprocessing import Pool, Manager
import queue
import asyncio
import os, pathlib, platform, time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp


WORKER_NUMBER = 5
GAMMA = 0.95
PPO2_EPSILON = 0.2
current_dir = pathlib.Path(__file__).parent.resolve()
os_type = platform.system() + '-' + platform.machine()



# Build actor model, we need two actor models,  one for prediction by workers; the other for training
def build_actor_model(trainable):
    return tf.keras.Sequential(
        [
            tf.keras.Input(type_spec=tf.RaggedTensorSpec(shape=[None, 8], dtype=tf.float32)),
            tf.keras.layers.Dense( 512, activation='relu', trainable=trainable),
            tf.keras.layers.LayerNormalization(trainable=trainable),
            tf.keras.layers.Dense( 512, activation='relu', trainable=trainable),
            tf.keras.layers.LayerNormalization(trainable=trainable),
            tf.keras.layers.Dense( 1024, activation='relu', trainable=trainable),
            tf.keras.layers.LayerNormalization(trainable=trainable),
            tf.keras.layers.Dense( 1024, activation='relu', trainable=trainable),
            tf.keras.layers.LayerNormalization(trainable=trainable),
            tf.keras.layers.Dense( 2048, activation='relu', trainable=trainable),
            tf.keras.layers.LayerNormalization(trainable=trainable),
            tf.keras.layers.Dense( 2048, activation='relu', trainable=trainable),
            tf.keras.layers.LayerNormalization(trainable=trainable),
            tf.keras.layers.Dense(2, activation='softmax', trainable=trainable)
        ]
    )

def build_critic_model():
    return tf.keras.Sequential(
        [
            tf.keras.Input(type_spec=tf.RaggedTensorSpec(shape=[None, 8], dtype=tf.float32)),
            tf.keras.layers.Dense( 1024, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense( 1024, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense( 1024, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense( 1024, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense( 1024, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(1, activation=None)
        ]
    )

######################################################################################
# entry of child worker process
def worker(id, c2p_queue, p2c_queue):
    print('Run worker (%s)...' % (os.getpid()))
    asyncio.run(async_worker(id, c2p_queue, p2c_queue))

# child process
async def async_worker(id, c2p_queue, p2c_queue):
    # run the game in chrome and collect data for training
    chromedriver_path = './{}/chromedriver'.format(os_type)
    print("Starting chromedriver at :" + chromedriver_path)
    service = Service(executable_path=chromedriver_path)
    driver = webdriver.Chrome(service=service)

    driver.get("file://" + str(current_dir / "JS-Flappy-Bird-master" / "index.html"))

    try:

        for episode in range(999999):
            driver.execute_script(" start();")

            actions = []
            rewards = []
            states = []
            log_probs = []

            total_rewards = 0

            #r = random.randint(1, 20)  # skip random steps
            #for _ in range(r):
            #    state_dic = driver.execute_script("return step(0);")

            state_dic = driver.execute_script("return step(0);")

            running = state_dic['running']
            state = [state_dic['to_gnd'], state_dic['to_roof'], state_dic['to_floor'], state_dic['to_start'], state_dic['next_frame_to_roof'], state_dic['next_frame_to_floor'], state_dic['speed'], state_dic['to_next_roof']]
        
            while running:
                action, log_prob = choose_action(id, c2p_queue, p2c_queue, state)
                actions.append(action)
                states.append(state)
                log_probs.append(log_prob)

                flap = action > 0
                state_dic = driver.execute_script( "return step({});".format("1" if flap else "0"))
                running = state_dic['running']
                reward = state_dic['reward']

                rewards.append(reward)
                total_rewards += reward
                new_state = [state_dic['to_gnd'], state_dic['to_roof'], state_dic['to_floor'], state_dic['to_start'],  state_dic['next_frame_to_roof'], state_dic['next_frame_to_floor'], state_dic['speed'], state_dic['to_next_roof']]
                
                state = new_state

            send_trajectory( id, c2p_queue, states, rewards, actions, log_probs)
            
            print("#{} Total rewards = {}".format(id, total_rewards))
 
    except Exception:
        driver.quit()

    driver.quit()


    

# call parent process to predict the action to take
def choose_action(id, c2p_queue, p2c_queue, state):
    c2p_queue.put({
        'id' : id,
        'choose_action' : state
    })
    reply = p2c_queue.get(block=True)
    return reply # (action, log_probability)


# send trajectory and states to parent process
def send_trajectory(id, c2p_queue, states, rewards, actions, log_probs):
    c2p_queue.put({
        'id' : id,
        'states' : states,
        'rewards' : rewards,
        'actions' : actions,
        'log_probs' : log_probs
    }, block=False)

   
def print_error(err):
    print(err)


######################################################################################
# entry of child trainer process
def trainer(id, c2p_queue, p2c_queue):
    print('Run trainer (%s)...' % (os.getpid()))
    actor_model = build_actor_model(True)
    actor_model.summary()
    actor_opt = tf.keras.optimizers.AdamW(learning_rate=0.000001)
    critic_opt = tf.keras.optimizers.AdamW(learning_rate=0.000005)
    critic_model = build_critic_model()
    critic_model.summary()

    try:
        actor_model.load_weights(str(current_dir) + "/checkpoints/ppo_discrete_actor/latest")
        critic_model.load_weights(str(current_dir) + "/checkpoints/ppo_discrete_critic/latest")
        tf.saved_model.save( actor_model, "./saved_model")
    except Exception:
        print("Failed to load actor_model or critic_model")

    count = 0
    while True:
        surrogate_trajectories = p2c_queue.get(block=True)
        for (_, states, rewards, actions, log_probs) in surrogate_trajectories:
            sum_reward = 0
            discounted_rewards = [] # Q : gain of each state
            rewards.reverse()  
            for r in rewards:
                sum_reward = r + GAMMA*sum_reward
                discounted_rewards.append(sum_reward)
            discounted_rewards.reverse()
            discounted_rewards = np.vstack(discounted_rewards)
            states = np.vstack(states)
            with tf.GradientTape() as critic_tape:
                values = critic_model(states, training=True)

                # discounted_rewards       values             advantages
                # [  1.81871276]       [-0.05801408]        [  1.8767267 ]
                # [  1.75378853]       [ 0.01909591]        [  1.7346926 ]
                #      ...         -        ...        =          ...
                # [-15.        ]       [-1.9742196 ]        [-13.025781  ]

                advantages = tf.subtract(discounted_rewards, values) # Advantage = Q(G) - V
                critic_loss = tf.reduce_mean(tf.square(advantages)) # MSE
            critic_grads = critic_tape.gradient(critic_loss, critic_model.trainable_variables) 
            critic_opt.apply_gradients( zip(critic_grads, critic_model.trainable_variables))

            with tf.GradientTape() as actor_tape:
                pairs = actor_model(states, training=True)
                #[[0.7888968  0.21110322]
                # ...
                #[0.79374665 0.20625333]]

                # e^[log(x)] = x;  x/y = e^[log(x)]/e^[log(y)] = e^[log(x)-log(y)]
                ratios = [ tf.exp(tf.math.log(pair[action] + 1e-10) - old_log_prob)  for pair, action, old_log_prob in zip(pairs, actions, log_probs)]

                # [1.0000008]      [-25.027773]    [-25.027794]
                # [1.00000014]  x  [-25.506348]  = [-25.506351]
                #      ...              ...      =      ...
                # [0.99999959]     [-26.04439 ]    [-26.044378]
                surrogate = tf.math.multiply( tf.expand_dims(ratios, axis=1), advantages )
                actor_loss = -tf.reduce_mean(tf.minimum( surrogate, tf.math.multiply( tf.clip_by_value(ratios, 1.-PPO2_EPSILON, 1.+PPO2_EPSILON), advantages )))
            
            actor_grads = actor_tape.gradient(actor_loss, actor_model.trainable_variables) 
            actor_opt.apply_gradients( zip(actor_grads, actor_model.trainable_variables))

        c2p_queue.put( { 'weights' : actor_model.get_weights() }, block=False)

        count += 1
        if count % 10:
            actor_model.save_weights(str(current_dir) + "/checkpoints/ppo_discrete_actor/latest", overwrite=True)
            critic_model.save_weights(str(current_dir) + "/checkpoints/ppo_discrete_critic/latest", overwrite=True)
                        


######################################################################################
# Parent process
if __name__=='__main__':
    surrogate_model = build_actor_model(False)

    try:
        surrogate_model.load_weights(str(current_dir) + "/checkpoints/ppo_discrete_actor/latest")
    except Exception:
        print("Failed to load surrogate_model")
    

    with Manager() as manager:
        c2p_queue = manager.Queue() # create a queue in parent process to receive data from child processes
        p2c_queue_dic = dict()
        print('Parent process %s.' % os.getpid())
        p = Pool(WORKER_NUMBER+1)
        for id in range(WORKER_NUMBER):
            p2c_queue = manager.Queue() # create a dedicated queue to send message
            p2c_queue_dic[id] = p2c_queue
            p.apply_async(func=worker, args=(id, c2p_queue, p2c_queue,), error_callback=print_error)

        p2t_queue = manager.Queue() # create a dedicated queue to send message to trainer
        p.apply_async(func=trainer, args=(id, c2p_queue, p2t_queue,), error_callback=print_error)

        start_time = time.time()
        while time.time() - start_time < 3600:

            states = []
            request_ids = []
            trajectories = []

            # communicate with child processes and collect states and results
            block=True
            while True:
                try:
                    msg = c2p_queue.get(block)
                    block = False
                    match msg:
                        # worker request to predict action
                        case { 'choose_action' : state, 'id' : id }:
                            states.append(state)
                            request_ids.append(id)
                            break

                        # worker reports a trajectory
                        case {
                                'id' : id,
                                'states' : states,
                                'rewards' : rewards,
                                'actions' : actions,
                                'log_probs' : log_probs
                            }:
                            assert len(states) == len(rewards), "states length must be the same as rewards"
                            trajectories.append( (id, states, rewards, actions, log_probs))
                            break

                        # trainer reports new weights
                        case { 'weights' : weights }:
                            surrogate_model.set_weights(weights)
                            break

                except queue.Empty: 
                    break #queue empty

            # call actor to predict
            if len(states) > 0:
                prob_batch = surrogate_model(np.array(states))
                for id, probabilities in zip(request_ids, prob_batch.numpy()):
                    dist = tfp.distributions.Categorical(probs=probabilities, dtype=tf.float32)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    p2c_queue_dic[id].put( (int(action.numpy()), log_prob.numpy()), block=False)

            if len(trajectories) > 0: # train
                p2t_queue.put( trajectories, block=False)
                trajectories = []


                
        p.close()
        #p.join()
        #print('All subprocesses done.')


