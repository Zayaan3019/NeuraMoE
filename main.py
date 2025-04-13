import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision
import time

# Enable mixed precision for UHAT-ASE training
mixed_precision.set_global_policy('mixed_float16')


##############################################
# Baseline ANN Model
##############################################

def build_ann_model(input_shape=(784,), num_classes=10):
    """
    A simple multi-layer perceptron as the baseline ANN.
    """
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


##############################################
# UHAT-ASE Model Components
##############################################

class MultiScaleAttention(keras.layers.Layer):
    """
    Multi-Scale Hierarchical Attention.
    Processes input at the original resolution and a downsampled scale.
    The coarse branch uses average pooling then upsamples back.
    """
    def __init__(self, d_model, pool_size=2, n_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.pool_size = pool_size
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            output_shape=d_model
        )
        self.upsample = keras.layers.UpSampling1D(size=pool_size)
        self.pool = keras.layers.AveragePooling1D(pool_size=pool_size)

    def call(self, x):
        # x: (batch, seq, d_model)
        fine_out = self.attn(query=x, key=x, value=x)
        x_pooled = self.pool(x)
        coarse = self.attn(query=x_pooled, key=x_pooled, value=x_pooled)
        coarse_up = self.upsample(coarse)
        coarse_up = coarse_up[:, :tf.shape(x)[1], :]
        return fine_out + coarse_up


class StateSpaceLayer(keras.layers.Layer):
    """
    Custom state-space layer inspired by recent SSM research.
    Uses a causal convolution and learnable transformation.
    """
    def __init__(self, d_state=16, d_conv=3, expand=2, **kwargs):
        super().__init__(**kwargs)
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

    def build(self, input_shape):
        # input_shape: (batch, seq, d_model)
        d_model = input_shape[-1]
        self.d_inner = int(self.expand * d_model)
        self.conv = keras.layers.Conv1D(
            filters=self.d_inner,
            kernel_size=self.d_conv,
            padding='causal',
            groups=d_model,
            use_bias=False
        )
        self.A = self.add_weight(name="A", shape=(self.d_inner, self.d_state),
                                 initializer='glorot_uniform', trainable=True)
        self.B = self.add_weight(name="B", shape=(self.d_inner, self.d_state),
                                 initializer='glorot_uniform', trainable=True)
        self.C = self.add_weight(name="C", shape=(self.d_state, self.d_inner),
                                 initializer='glorot_uniform', trainable=True)
        self.out_proj = keras.layers.Dense(d_model, dtype='float32')
        super().build(input_shape)

    def call(self, x):
        # x: (batch, seq, d_model)
        residual = x
        x = self.conv(x)  # (batch, seq, d_inner)
        A = tf.exp(self.A)
        B = tf.nn.silu(self.B)
        C = tf.nn.silu(self.C)
        x1 = tf.linalg.matmul(x, A)  # (batch, seq, d_state)
        x2 = tf.linalg.matmul(x, B)  # (batch, seq, d_state)
        x = x1 * x2                # (batch, seq, d_state)
        x = tf.linalg.matmul(x, C)  # (batch, seq, d_inner)
        return self.out_proj(x) + tf.cast(residual, tf.float32)


class AdaptiveSparseMoE(keras.layers.Layer):
    """
    Adaptive Sparse Mixture-of-Experts (AS-MoE).
    Uses a gating mechanism to select top-k experts per token.
    """
    def __init__(self, d_model, moe_capacity=4, k=2, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.moe_capacity = moe_capacity
        self.k = k

    def build(self, input_shape):
        self.experts = []
        for i in range(self.moe_capacity):
            expert = keras.Sequential([
                keras.layers.Dense(self.d_model * 4, activation='swish'),
                keras.layers.Dense(self.d_model)
            ], name=f"expert_{i}")
            self.experts.append(expert)
        self.gate = keras.layers.Dense(self.moe_capacity)
        super().build(input_shape)

    def call(self, x):
        # x: (batch, seq, d_model)
        gate_logits = self.gate(x)
        topk, indices = tf.math.top_k(gate_logits, k=self.k)
        # Process each expert
        expert_outputs = []
        for i in range(self.moe_capacity):
            expert_out = self.experts[i](x)
            expert_outputs.append(expert_out)
        expert_stack = tf.stack(expert_outputs, axis=-1)  # (batch, seq, d_model, moe_capacity)
        one_hot = tf.reduce_sum(tf.one_hot(indices, depth=self.moe_capacity, dtype=x.dtype), axis=-2)
        gate_mask_exp = tf.expand_dims(one_hot, axis=2)
        moe_out = tf.reduce_sum(expert_stack * gate_mask_exp, axis=-1)  # (batch, seq, d_model)
        return tf.cast(moe_out, tf.float32)


class UHATASEBlock(keras.layers.Layer):
    """
    A single block that combines multi-scale attention, state-space, and AS-MoE.
    """
    def __init__(self, d_model, n_heads=8, pool_size=2, d_state=16, moe_capacity=4, k=2, **kwargs):
        super().__init__(**kwargs)
        self.multi_scale_attn = MultiScaleAttention(d_model, pool_size=pool_size, n_heads=n_heads)
        self.state_space = StateSpaceLayer(d_state=d_state)
        self.adaptive_moe = AdaptiveSparseMoE(d_model, moe_capacity=moe_capacity, k=k)
        self.norm = keras.layers.LayerNormalization()

    def call(self, x):
        x_attn = self.multi_scale_attn(x)
        x_ss = self.state_space(x)
        scaler = tf.constant(0.5, dtype=tf.float32)
        blended = scaler * tf.cast(x_attn, tf.float32) + scaler * x_ss
        moe_out = tf.cast(self.adaptive_moe(blended), tf.float32)
        return self.norm(blended + moe_out)


class UHATASEModel(keras.Model):
    """
    UHAT-ASE Model: Stacks multiple UHATASEBlocks and applies a classification head.
    """
    def __init__(self, num_classes=10, num_blocks=4, d_model=256, n_heads=8, pool_size=2,
                 d_state=16, moe_capacity=4, k=2, **kwargs):
        super().__init__(**kwargs)
        self.embed = keras.layers.Dense(d_model)
        self.blocks = [UHATASEBlock(d_model, n_heads=n_heads, pool_size=pool_size,
                                    d_state=d_state, moe_capacity=moe_capacity, k=k)
                       for _ in range(num_blocks)]
        self.pool = keras.layers.GlobalAveragePooling1D()
        self.classifier = keras.layers.Dense(num_classes, activation='softmax', dtype='float32')

    def call(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        return self.classifier(x)


##############################################
# Data Preparation
##############################################
def prepare_mnist_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Normalize and reshape:
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    # For ANN, flatten the images (28x28 => 784)
    x_train_flat = x_train.reshape(-1, 784)
    x_test_flat = x_test.reshape(-1, 784)
    # For UHAT-ASE, treat each image as a sequence of 784 timesteps with 1 feature each.
    x_train_seq = x_train.reshape(-1, 784, 1)
    x_test_seq = x_test.reshape(-1, 784, 1)
    return (x_train_flat, y_train, x_test_flat, y_test), (x_train_seq, y_train, x_test_seq, y_test)


##############################################
# Training and Evaluation Functions
##############################################
def compile_and_train(model, x_train, y_train, x_val, y_val, batch_size=256, epochs=10):
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    start = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(patience=3),
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ],
        verbose=2
    )
    end = time.time()
    training_time = end - start
    return history, training_time


def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return loss, accuracy


##############################################
# Main: Train and Compare the Two Models
##############################################
if __name__ == "__main__":
    # Prepare data.
    (x_train_flat, y_train_flat, x_test_flat, y_test_flat), (x_train_seq, y_train_seq, x_test_seq, y_test_seq) = prepare_mnist_data()

    # Split training data into training and validation sets (80/20 split).
    val_split = 0.2
    num_val_flat = int(len(x_train_flat) * val_split)
    num_val_seq = int(len(x_train_seq) * val_split)

    x_train_ann = x_train_flat[num_val_flat:]
    y_train_ann = y_train_flat[num_val_flat:]
    x_val_ann   = x_train_flat[:num_val_flat]
    y_val_ann   = y_train_flat[:num_val_flat]

    x_train_uhat = x_train_seq[num_val_seq:]
    y_train_uhat = y_train_seq[num_val_seq:]
    x_val_uhat   = x_train_seq[:num_val_seq]
    y_val_uhat   = y_train_seq[:num_val_seq]

    # Build baseline ANN model.
    ann_model = build_ann_model(input_shape=(784,), num_classes=10)
    ann_model.summary()

    # Build UHAT-ASE model.
    uhat_model = UHATASEModel(num_classes=10, num_blocks=4, d_model=256, n_heads=8,
                              pool_size=2, d_state=16, moe_capacity=4, k=2)
    # For UHAT-ASE, build with the input shape of (None, 784, 1)
    uhat_model.build(input_shape=(None, 784, 1))
    uhat_model.summary()

    # Train the baseline ANN.
    print("\nTraining Baseline ANN Model...")
    ann_history, ann_training_time = compile_and_train(ann_model, x_train_ann, y_train_ann, x_val_ann, y_val_ann)
    ann_loss, ann_acc = evaluate_model(ann_model, x_test_flat, y_test_flat)
    print(f"Baseline ANN Test Accuracy: {ann_acc:.4f}, Training Time: {ann_training_time:.2f} sec")

    # Train the UHAT-ASE model.
    print("\nTraining UHAT-ASE Model...")
    uhat_history, uhat_training_time = compile_and_train(uhat_model, x_train_uhat, y_train_uhat, x_val_uhat, y_val_uhat)
    uhat_loss, uhat_acc = evaluate_model(uhat_model, x_test_seq, y_test_seq)
    print(f"UHAT-ASE Test Accuracy: {uhat_acc:.4f}, Training Time: {uhat_training_time:.2f} sec")

    ##############################################
    # Plotting Comparison
    ##############################################
    models_names = ['Baseline ANN', 'UHAT-ASE']
    test_accuracies = [ann_acc, uhat_acc]
    training_times = [ann_training_time, uhat_training_time]

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.bar(models_names, test_accuracies, color=['blue', 'green'])
    plt.title('Test Accuracy Comparison')
    plt.ylabel('Accuracy')
    for i, v in enumerate(test_accuracies):
        plt.text(i, v + 0.005, f"{v:.3f}", ha='center')

    plt.subplot(1, 2, 2)
    plt.bar(models_names, training_times, color=['orange', 'purple'])
    plt.title('Training Time Comparison (sec)')
    plt.ylabel('Time (sec)')
    for i, v in enumerate(training_times):
        plt.text(i, v + 1, f"{v:.1f}", ha='center')

    plt.tight_layout()
    plt.show()

    ##############################################
    # Plot Training Curves for UHAT-ASE and ANN
    ##############################################
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(ann_history.history['accuracy'], label='ANN Train')
    plt.plot(ann_history.history['val_accuracy'], label='ANN Val')
    plt.title('Baseline ANN Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(uhat_history.history['accuracy'], label='UHAT-ASE Train')
    plt.plot(uhat_history.history['val_accuracy'], label='UHAT-ASE Val')
    plt.title('UHAT-ASE Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
