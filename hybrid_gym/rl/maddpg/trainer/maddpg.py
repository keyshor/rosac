import numpy as np
import tensorflow as tf

from hybrid_gym.rl.maddpg.common import tf_util as U
from hybrid_gym.rl.maddpg.common.distributions import make_pdtype
from hybrid_gym.rl.maddpg.trainer.replay_buffer import ReplayBuffer

DEFAULT_TAU = 1.0 - 1e-2


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]


def make_update_exp(vals, target_vals, sess, polyak=DEFAULT_TAU):
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name),
                               sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], sess=sess, updates=[expression])


def p_train(make_obs_ph_n, act_space_n, p_func, q_func, optimizer, sess,
            grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer",
            reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = make_pdtype(act_space_n)

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = act_pdtype_n.sample_placeholder([None], name="action")
        p_input = obs_ph_n

        p = p_func(p_input, int(act_pdtype_n.param_shape()[0]),
                   scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n.pdfromflat(p)

        act_sample = act_pd.sample()
        act_det = act_pd.det_sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_pd.sample()
        q_input = tf.concat([obs_ph_n, act_input_n], 1)

        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:, 0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + 0.01 * p_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=[obs_ph_n, act_ph_n], outputs=loss,
                           sess=sess, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n], outputs=act_sample, sess=sess)
        det_action = U.function(inputs=[obs_ph_n], outputs=act_det, sess=sess)
        p_values = U.function([obs_ph_n], p, sess=sess)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n.param_shape()[0]),
                          scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars, sess)

        target_act_sample = act_pdtype_n.pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n], outputs=target_act_sample, sess=sess)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act,
                                             'det_action': det_action}


def p_copy(make_obs_ph_n, act_space_n, p_func, sess, num_units=64, scope="copy",
           reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = make_pdtype(act_space_n)

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        p_input = obs_ph_n

        p_copy = p_func(p_input, int(act_pdtype_n.param_shape()[0]),
                        scope="p_copy", num_units=num_units)

        # wrap parameters in distribution
        act_pd = act_pdtype_n.pdfromflat(p_copy)

        act_sample = act_pd.sample()
        act_det = act_pd.det_sample()

        act = U.function(inputs=[obs_ph_n], outputs=act_sample, sess=sess)
        det_action = U.function(inputs=[obs_ph_n], outputs=act_det, sess=sess)

        # target network
        p_copy_vars = U.scope_vars(U.absolute_scope_name("p_copy"))
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        update_copy_p = make_update_exp(p_func_vars, p_copy_vars, sess, polyak=0.)

        return act, det_action, update_copy_p


def q_train(make_obs_ph_n, act_space_n, q_func, optimizer, sess, grad_norm_clipping=None,
            local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = make_pdtype(act_space_n)

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = act_pdtype_n.sample_placeholder([None], name="action")
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat([obs_ph_n, act_ph_n], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:, 0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))  # noqa
        loss = q_loss  # + 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=[obs_ph_n] + [act_ph_n] +
                           [target_ph], outputs=loss, sess=sess, updates=[optimize_expr])
        q_values = U.function([obs_ph_n, act_ph_n], q, sess=sess)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:, 0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars, sess)

        target_q_values = U.function([obs_ph_n, act_ph_n], target_q, sess=sess)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


class MADDPGAgentTrainer:
    def __init__(self, name, model, obs_shape_n, act_space_n,
                 args, sess, local_q_func=False,
                 adversary=False, bonus=50.):
        self.name = name
        self.n = len(obs_shape_n)
        self.args = args
        self.sess = sess
        self.adversary = adversary
        self.bonus = bonus
        obs_ph_n = U.BatchInput(obs_shape_n, name="observation_"+self.name).get()

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            sess=self.sess
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            sess=self.sess
        )

        self.act_copy, self.det_act_copy, self.copy_update_p = p_copy(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_func=model,
            num_units=args.num_units,
            sess=self.sess
        )

        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs, copy=False):
        if copy:
            return self.act_copy(obs[None])[0]
        else:
            return self.act(obs[None])[0]

    def deterministic_action(self, obs, copy=False):
        if copy:
            return self.det_act_copy(obs[None])[0]
        else:
            return self.p_debug['det_action'](obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, success):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done), float(success))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, other_agents, t):
        # other_agents is adv agent for option learners and list of option learners for adv agents
        if len(self.replay_buffer) < self.max_replay_buffer_len:
            # replay buffer is not large enough
            return
        if not t % 10 == 0:
            # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)

        # collect replay sample from all agents
        index = self.replay_sample_index
        obs, act, rew, obs_next, done, success = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for _ in range(num_sample):
            if self.adversary:
                for i in range(len(act)):
                    target_act_next = other_agents[i].p_debug['target_act'](obs_next[:, i, :])
                    target_q_next = other_agents[i].q_debug['target_q_values'](
                        obs_next[:, i, :], target_act_next)
                    target_q -= act[:, i] * (rew + target_q_next)
            else:
                target_act_next = self.p_debug['target_act'](obs_next)
                adv_act_next = other_agents.p_debug['target_act'](obs_next)
                target_q_next = self.q_debug['target_q_values'](obs_next, target_act_next)
                adv_target_next = -other_agents.q_debug['target_q_values'](obs_next, adv_act_next)
                target_q += (rew + self.args.gamma * (1.0 - done) * (1.0 - success) * target_q_next
                             + success * (adv_target_next + self.bonus))
        target_q /= num_sample
        q_loss = self.q_train(obs, act, target_q)

        # train p network
        p_loss = self.p_train(obs, act)

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.std(target_q)]
