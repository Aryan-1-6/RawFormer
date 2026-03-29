import cupy as np


class Loss_CrossCategoricalEntropy:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss     = np.mean(sample_losses)
        return data_loss

    def forward(self, y_pred, y_true):
        y_pred_clipped    = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = []

        # Case 1: (batch, seq_len, vocab)
        if len(y_pred.shape) == 3:
            batch_size, seq_len, vocab_size = y_pred.shape

            if len(y_true.shape) == 2:        # integer labels
                correct_confidences = y_pred_clipped[
                    np.arange(batch_size)[:, None],
                    np.arange(seq_len)[None, :],
                    y_true
                ]
            elif len(y_true.shape) == 3:      # one-hot labels
                correct_confidences = np.sum(y_pred_clipped * y_true, axis=-1)

        # Case 2: (batch, vocab)
        elif len(y_pred.shape) == 2:
            if len(y_true.shape) == 1:
                correct_confidences = y_pred_clipped[np.arange(len(y_pred)), y_true]
            elif len(y_true.shape) == 2:
                correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        else:
            raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")

        negative_log_likelihoods = -np.log(correct_confidences + 1e-10)
        return negative_log_likelihoods

    def backward(self, y_pred, y_true):
        B, T, V = y_pred.shape
        self.dinputs = y_pred.astype(np.float32).copy()

        # Combined softmax + cross-entropy gradient: y_pred - one_hot(y_true)
        self.dinputs[
            np.arange(B)[:, None],
            np.arange(T)[None, :],
            y_true
        ] -= 1

        self.dinputs /= (B * T)
        return self.dinputs