import cupy as np


class Optimizer_SGD:
    def __init__(self, learning_rate=0.00001):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases  -= self.learning_rate * layer.dbiases

# Primary optimizer
class OptimizerAdam:
    def __init__(self, learning_rate=0.0003, beta1=0.9, beta2=0.98,
                 epsilon=1e-8, warmup_steps=200):
        self.peak_lr      = learning_rate
        self.learning_rate = learning_rate
        self.beta1        = beta1
        self.beta2        = beta2
        self.epsilon      = epsilon
        self.warmup_steps = warmup_steps
        self.t            = 0

    def pre_update(self):
        self.t += 1
        # Linear warmup then constant LR
        if self.t < self.warmup_steps:
            self.learning_rate = self.peak_lr * (self.t / self.warmup_steps)
        else:
            self.learning_rate = self.peak_lr

    def update_params(self, layer):

        # ---- Dense layers ----
        if hasattr(layer, 'weights'):
            if not hasattr(layer, 'm_w'):
                layer.m_w = np.zeros_like(layer.weights)
                layer.v_w = np.zeros_like(layer.weights)
                layer.m_b = np.zeros_like(layer.biases)
                layer.v_b = np.zeros_like(layer.biases)

            # Optional gradient clipping (uncomment if gradients explode)
            # np.clip(layer.dweights, -1.0, 1.0, out=layer.dweights)
            # np.clip(layer.dbiases,  -1.0, 1.0, out=layer.dbiases)

            layer.m_w = self.beta1 * layer.m_w + (1 - self.beta1) * layer.dweights
            layer.v_w = self.beta2 * layer.v_w + (1 - self.beta2) * (layer.dweights ** 2)
            layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * layer.dbiases
            layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * (layer.dbiases ** 2)

            m_w_hat = layer.m_w / (1 - self.beta1 ** self.t)
            v_w_hat = layer.v_w / (1 - self.beta2 ** self.t)
            m_b_hat = layer.m_b / (1 - self.beta1 ** self.t)
            v_b_hat = layer.v_b / (1 - self.beta2 ** self.t)

            layer.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            layer.biases  -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        # ---- LayerNorm ----
        if hasattr(layer, 'gamma'):
            if not hasattr(layer, 'm_g'):
                layer.m_g    = np.zeros_like(layer.gamma)
                layer.v_g    = np.zeros_like(layer.gamma)
                layer.m_beta = np.zeros_like(layer.beta)
                layer.v_beta = np.zeros_like(layer.beta)

            layer.m_g    = self.beta1 * layer.m_g    + (1 - self.beta1) * layer.dgamma
            layer.v_g    = self.beta2 * layer.v_g    + (1 - self.beta2) * (layer.dgamma ** 2)
            layer.m_beta = self.beta1 * layer.m_beta + (1 - self.beta1) * layer.dbeta
            layer.v_beta = self.beta2 * layer.v_beta + (1 - self.beta2) * (layer.dbeta ** 2)

            m_g_hat    = layer.m_g    / (1 - self.beta1 ** self.t)
            v_g_hat    = layer.v_g    / (1 - self.beta2 ** self.t)
            m_beta_hat = layer.m_beta / (1 - self.beta1 ** self.t)
            v_beta_hat = layer.v_beta / (1 - self.beta2 ** self.t)

            layer.gamma -= self.learning_rate * m_g_hat    / (np.sqrt(v_g_hat)    + self.epsilon)
            layer.beta  -= self.learning_rate * m_beta_hat / (np.sqrt(v_beta_hat) + self.epsilon)

    def update_params_embeddings(self, model):
        if not hasattr(model, 'embedding_momentums'):
            model.embedding_momentums = np.zeros_like(model.embeddings)
            model.embedding_cache     = np.zeros_like(model.embeddings)

        model.embedding_momentums = (
            self.beta1 * model.embedding_momentums +
            (1 - self.beta1) * model.dembeddings
        )
        model.embedding_cache = (
            self.beta2 * model.embedding_cache +
            (1 - self.beta2) * (model.dembeddings ** 2)
        )

        m_hat = model.embedding_momentums / (1 - self.beta1 ** self.t)
        v_hat = model.embedding_cache     / (1 - self.beta2 ** self.t)

        model.embeddings -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)