import numpy as np
import tensorflow as tf

from hybrid_gym.rl.maddpg.common import tf_util as U
from hybrid_gym.rl.maddpg.common.distributions import make_pdtype
from hybrid_gym.rl.maddpg.trainer.replay_buffer import ReplayBuffer

DEFAULT_TAU = 1.0 - 1e-2


def make_update_exp(vals, target_vals, sess, polyak=DEFAULT_TAU):
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name),
                               sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], sess=sess, updates=[expression])


def p_train(obs_ph, act_space, p_func, q_func, optimizer, sess,
            grad_norm_clipping=None, num_units=64, scope="trainer",
            reuse=None, num_modes=1):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype = make_pdtype(act_space)

        # set up placeholders
        act_ph = act_pdtype.sample_placeholder([None], name="action")
        adv_act_ph = tf.placeholder(tf.int32, [None], name="adv_action")

        p = p_func(obs_ph, int(act_pdtype.param_shape()[0]),
                   scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype.pdfromflat(p)

        act_sample = act_pd.sample()
        act_det = act_pd.mode()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input = act_pd.sample()
        q_input = tf.concat([obs_ph, act_input], 1)
        q_full = q_func(q_input, num_modes, scope="q_func", reuse=True, num_units=num_units)
        q = tf.gather(q_full, tf.reshape(adv_act_ph, [-1, 1]), axis=1, batch_dims=1)
        q = tf.reshape(q, [-1])

        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=[obs_ph, act_ph, adv_act_ph], outputs=loss,
                           sess=sess, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph], outputs=act_sample, sess=sess)
        det_act = U.function(inputs=[obs_ph], outputs=act_det, sess=sess)
        p_values = U.function([obs_ph], p, sess=sess)

        # target network
        target_p = p_func(obs_ph, int(act_pdtype.param_shape()[0]),
                          scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars, sess)

        target_act_sample = act_pdtype.pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph], outputs=target_act_sample, sess=sess)

        return act, train, update_target_p, {
            'p_values': p_values, 'target_act': target_act, 'det_action': det_act}


def p_train_adv(obs_ph, act_space, p_func, q_func, optimizer, sess,
                grad_norm_clipping=None, num_units=64, scope="trainer",
                reuse=None, num_modes=1):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype = make_pdtype(act_space)

        # set up placeholders
        act_ph = act_pdtype.sample_placeholder([None], name="action")
        adv_act_ph = tf.placeholder(tf.int32, [None], name="adv_action")

        p = p_func(obs_ph, num_modes, activation_fn=tf.nn.softmax,
                   scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        adv_act_pd = tf.distributions.Categorical(probs=p)
        adv_act_sample = adv_act_pd.sample()
        adv_act_det = U.argmax(p, axis=1)

        q_input = tf.concat([obs_ph, act_ph], 1)
        q_full = q_func(q_input, num_modes, scope="q_func", reuse=True, num_units=num_units)
        q = tf.reduce_sum(tf.multiply(q_full, p), 1)

        loss = -tf.reduce_mean(q)

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=[obs_ph, act_ph, adv_act_ph], outputs=loss,
                           sess=sess, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph], outputs=adv_act_sample, sess=sess)
        det_act = U.function(inputs=[obs_ph], outputs=adv_act_det, sess=sess)
        p_values = U.function([obs_ph], p, sess=sess)

        # target network
        target_p = p_func(obs_ph, num_modes, activation_fn=tf.nn.softmax,
                          scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars, sess)

        target_act_sample = tf.distributions.Categorical(probs=target_p).sample()
        target_act = U.function(inputs=[obs_ph], outputs=target_act_sample, sess=sess)

        return act, train, update_target_p, {
            'p_values': p_values, 'target_act': target_act, 'det_action': det_act}


def q_train(obs_ph, act_space, q_func, optimizer, sess, grad_norm_clipping=None,
            scope="trainer", reuse=None, num_units=64, num_modes=1):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype = make_pdtype(act_space)

        # set up placeholders
        act_ph = act_pdtype.sample_placeholder([None], name="action")
        adv_act_ph = tf.placeholder(tf.int32, [None], name="adv_action")
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat([obs_ph, act_ph], 1)
        q_full = q_func(q_input, num_modes, scope="q_func", num_units=num_units)
        q = tf.gather(q_full, tf.reshape(adv_act_ph, [-1, 1]), axis=1, batch_dims=1)
        q = tf.reshape(q, [-1])
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))  # noqa
        loss = q_loss + 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=[obs_ph, act_ph, adv_act_ph, target_ph],
                           outputs=loss, sess=sess, updates=[optimize_expr])
        q_values = U.function([obs_ph, act_ph, adv_act_ph], q, sess=sess)

        # target network
        target_q_full = q_func(q_input, num_modes, scope="target_q_func", num_units=num_units)
        target_q = tf.gather(target_q_full, tf.reshape(adv_act_ph, [-1, 1]), axis=1, batch_dims=1)
        target_q = tf.reshape(target_q, [-1])
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars, sess)

        target_q_values = U.function([obs_ph, act_ph, adv_act_ph], target_q, sess=sess)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


class MADDPGAgentTrainer:
    def __init__(self, name, model, obs_shape, act_space,
                 args, sess, adv=False, num_modes=1):
        self.name = name
        self.args = args
        self.sess = sess
        self.num_modes = num_modes
        self.adv = adv
        obs_ph = U.BatchInput(obs_shape, name="observation").get()

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            obs_ph=obs_ph,
            act_space=act_space,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            scope=self.name,
            grad_norm_clipping=0.5,
            num_units=args.num_units,
            sess=self.sess,
            num_modes=self.num_modes
        )

        if adv:
            self.act, self.p_train, self.p_update, self.p_debug = p_train_adv(
                obs_ph=obs_ph,
                act_space=act_space,
                p_func=model,
                q_func=model,
                optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
                scope=self.name,
                grad_norm_clipping=0.5,
                num_units=args.num_units,
                sess=self.sess,
                num_modes=self.num_modes
            )

        else:
            self.act, self.p_train, self.p_update, self.p_debug = p_train(
                obs_ph=obs_ph,
                act_space=act_space,
                p_func=model,
                q_func=model,
                optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
                scope=self.name,
                grad_norm_clipping=0.5,
                num_units=args.num_units,
                sess=self.sess,
                num_modes=self.num_modes
            )

        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * 10
        self.replay_sample_index = None

    def action(self, obs):
        return self.act(obs[None])[0]

    def det_action(self, obs):
        return self.p_debug['det_action'](obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agent, adv_agent, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len:
            # replay buffer is not large enough
            return
        if not t % 100 == 0:
            # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents

        index = self.replay_sample_index
        obs, act, rew, obs_next, done = agent.replay_buffer.sample_index(index)
        _, adv_act, adv_rew, _, _ = adv_agent.replay_buffer.sample_index(index)
        if self.adv:
            rew = adv_rew

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act = agent.p_debug['target_act'](obs_next)
            target_adv_act = adv_agent.p_debug['target_act'](obs_next)
            target_q_next = self.q_debug['target_q_values'](obs_next, target_act, target_adv_act)
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(obs, act, adv_act, target_q)

        # train p network
        p_loss = self.p_train(obs, act, adv_act)

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.std(target_q)]
