import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
from tensorflow.keras import layers, Model
import time

class KolmogorovLayer(layers.Layer):
    """
    Implementation of Kolmogorov-Arnold Network (KAN) layer
    Inspired by research showing KANs can learn complex functions with fewer parameters
    """
    def __init__(self, units, spline_order=3, num_bases=8, **kwargs):
        super(KolmogorovLayer, self).__init__(**kwargs)
        self.units = units
        self.spline_order = spline_order
        self.num_bases = num_bases
        
    def build(self, input_shape):
        # Input projection weights
        self.projection = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            name='projection'
        )
        
        # B-spline basis control points
        self.control_points = self.add_weight(
            shape=(self.num_bases, self.units),
            initializer='glorot_uniform',
            name='control_points'
        )
        
        # Learnable scaling factors
        self.scaling = self.add_weight(
            shape=(self.units,),
            initializer='ones',
            name='scaling'
        )
        
        self.built = True
        
    def call(self, inputs):
        # Project inputs to KAN space
        x = tf.matmul(inputs, self.projection)
        
        # Normalize to [0,1] for spline interpolation
        x_norm = tf.nn.sigmoid(x)
        
        # Compute B-spline basis functions
        basis_values = self._compute_bspline_basis(x_norm)
        
        # Combine with control points
        output = tf.matmul(basis_values, self.control_points)
        
        # Apply scaling
        output = output * self.scaling
        
        return output
    
    def _compute_bspline_basis(self, x):
        """Approximate B-spline basis computation"""
        # Create normalized positions for the bases
        knots = tf.linspace(0.0, 1.0, self.num_bases)
        knots = tf.reshape(knots, [1, -1])
        
        # Compute simple basis functions (approximating B-splines)
        # Using squared exponential kernel for simplicity
        x_expanded = tf.expand_dims(x, axis=-1)
        dist = (x_expanded - knots) ** 2
        basis = tf.exp(-dist / (2.0 * (1.0/self.num_bases)**2))
        
        # Normalize the basis functions
        basis = basis / (tf.reduce_sum(basis, axis=-1, keepdims=True) + 1e-6)
        
        return basis

class MixtureOfExpertsLayer(layers.Layer):
    """
    Implements a Mixture of Experts layer with Gumbel-Softmax gating
    Inspired by "GG MoE vs. MLP on Tabular Data" research
    """
    def __init__(self, num_experts, expert_units, gating_temperature=1.0, **kwargs):
        super(MixtureOfExpertsLayer, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.expert_units = expert_units
        self.gating_temperature = gating_temperature
        
    def build(self, input_shape):
        # Create experts
        self.experts = []
        for i in range(self.num_experts):
            self.experts.append(keras.Sequential([
                layers.Dense(self.expert_units, activation='swish'),
                KolmogorovLayer(self.expert_units)
            ]))
        
        # Gating network
        self.gating = layers.Dense(self.num_experts)
        
        self.built = True
        
    def call(self, inputs, training=None):
        # Get expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(inputs))
        
        # Stack expert outputs along a new axis
        expert_outputs = tf.stack(expert_outputs, axis=1)  # [batch, num_experts, expert_units]
        
        # Compute gating weights with Gumbel-Softmax for training
        logits = self.gating(inputs)  # [batch, num_experts]
        
        if training:
            # Gumbel-Softmax with straight-through estimator for discrete selection
            u = tf.random.uniform(tf.shape(logits), minval=0, maxval=1)
            gumbel = -tf.math.log(-tf.math.log(u + 1e-10) + 1e-10)
            gating_weights = tf.nn.softmax((logits + gumbel) / self.gating_temperature, axis=-1)
            
            # Straight-through estimation
            gating_weights_hard = tf.one_hot(tf.argmax(gating_weights, axis=-1), self.num_experts)
            gating_weights = tf.stop_gradient(gating_weights_hard - gating_weights) + gating_weights
        else:
            # For inference, just use argmax
            gating_weights = tf.one_hot(tf.argmax(logits, axis=-1), self.num_experts)
        
        # Apply gating weights to expert outputs
        gating_weights = tf.expand_dims(gating_weights, axis=-1)  # [batch, num_experts, 1]
        output = tf.reduce_sum(expert_outputs * gating_weights, axis=1)  # [batch, expert_units]
        
        return output

class SelfAttentionBlock(layers.Layer):
    """
    Multi-head self-attention block for capturing complex data relationships
    """
    def __init__(self, embed_dim, num_heads=4, key_dim=32, dropout_rate=0.1, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=key_dim,
            dropout=dropout_rate
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential([
            layers.Dense(embed_dim * 2, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim)
        ])
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=None):
        # Self-attention with residual connection
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class NeuraMoETransformer(Model):
    """
    NeuraMoE Transformer: A revolutionary neural architecture combining Kolmogorov-Arnold 
    Networks, Mixture of Experts, and Self-Attention mechanisms
    """
    def __init__(self, 
                 input_dim,
                 num_classes=None,
                 num_experts=4,
                 expert_units=64,
                 attention_heads=4,
                 kan_units=32,
                 transformer_blocks=2,
                 dropout_rate=0.2,
                 use_adaptive_computation=True,
                 task_type='classification',
                 **kwargs):
        super(NeuraMoETransformer, self).__init__(**kwargs)
        
        self.task_type = task_type
        self.use_adaptive_computation = use_adaptive_computation
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Input embedding with KAN
        self.input_embedding = keras.Sequential([
            layers.Dense(kan_units * 2),
            KolmogorovLayer(kan_units)
        ])
        
        # Positional encoding for sequential data
        self.positional_encoding = self.get_positional_encoding(kan_units, 5000)
        
        # Transformer blocks
        self.transformer_blocks = []
        for _ in range(transformer_blocks):
            self.transformer_blocks.append(SelfAttentionBlock(
                embed_dim=kan_units,
                num_heads=attention_heads,
                dropout_rate=dropout_rate
            ))
        
        # Mixture of Experts layers
        self.moe_layer = MixtureOfExpertsLayer(
            num_experts=num_experts,
            expert_units=expert_units
        )
        
        # Adaptive computation halting mechanism
        if use_adaptive_computation:
            self.halting_unit = layers.Dense(1, activation='sigmoid')
        
        # Final output layers
        self.pre_output = keras.Sequential([
            layers.Dense(kan_units, activation='swish'),
            layers.Dropout(dropout_rate)
        ])
        
        # Task-specific output layer
        if task_type == 'classification':
            if num_classes is None or num_classes == 2:  # Binary classification
                self.output_layer = layers.Dense(1, activation='sigmoid')
            else:  # Multi-class classification
                self.output_layer = layers.Dense(num_classes, activation='softmax')
        else:  # Regression
            self.output_layer = layers.Dense(1, activation='linear')
    
    def call(self, inputs, training=None):
        # Input embedding
        x = self.input_embedding(inputs)
        
        # Add positional encoding for sequential data support
        # For tabular data, this has minimal effect but enables sequential processing
        x = x + self.positional_encoding[:, :tf.shape(x)[1], :]
        
        # Process through transformer blocks with adaptive computation
        transformer_outputs = []
        halting_probs = []
        
        for block in self.transformer_blocks:
            x = block(x, training=training)
            transformer_outputs.append(x)
            
            # Compute halting probability if using adaptive computation
            if self.use_adaptive_computation and training:
                halt_prob = self.halting_unit(x)
                halting_probs.append(halt_prob)
                
                # Apply adaptive computation during training with stochastic depth
                if len(halting_probs) > 1:
                    random_continue = tf.cast(
                        tf.random.uniform(tf.shape(halt_prob)) > halt_prob,
                        tf.float32
                    )
                    # Apply stochastic path dropping
                    x = x * random_continue
        
        # Expert routing
        x = self.moe_layer(x, training=training)
        
        # Final processing
        x = self.pre_output(x)
        outputs = self.output_layer(x)
        
        # During training, return outputs and halting probabilities
        if training and self.use_adaptive_computation:
            return outputs, halting_probs
        
        return outputs
    
    def get_positional_encoding(self, d_model, max_seq_len):
        """Create sinusoidal positional encoding"""
        positional_encoding = np.zeros((1, max_seq_len, d_model))
        positions = np.arange(0, max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        positional_encoding[0, :, 0::2] = np.sin(positions * div_term)
        positional_encoding[0, :, 1::2] = np.cos(positions * div_term)
        
        return tf.constant(positional_encoding, dtype=tf.float32)
    
    def compute_output_shape(self, input_shape):
        if self.task_type == 'classification':
            if self.num_classes is None or self.num_classes == 2:
                return (input_shape[0], 1)
            else:
                return (input_shape[0], self.num_classes)
        else:  # Regression
            return (input_shape[0], 1)

    def compile(self, optimizer='adam', loss=None, metrics=None, **kwargs):
        """Custom compile method to handle adaptive computation loss"""
        # Default loss based on task type
        if loss is None:
            if self.task_type == 'classification':
                if self.num_classes is None or self.num_classes == 2:
                    loss = 'binary_crossentropy'
                else:
                    loss = 'categorical_crossentropy'
            else:  # Regression
                loss = 'mean_squared_error'
        
        # Default metrics based on task type
        if metrics is None:
            if self.task_type == 'classification':
                metrics = ['accuracy']
            else:  # Regression
                metrics = ['mae']
        
        # Define our custom loss wrapper for adaptive computation
        if self.use_adaptive_computation:
            original_loss = loss
            
            def adaptive_loss(y_true, y_pred_and_halt):
                y_pred, halting_probs = y_pred_and_halt
                
                # Main task loss
                task_loss = tf.keras.losses.get(original_loss)(y_true, y_pred)
                
                # Adaptive computation loss - encourage efficient computation
                halt_loss = 0.0
                cumulative_halt = 0.0
                
                for i, p in enumerate(halting_probs):
                    # Ponder cost increases with depth
                    layer_cost = 0.01 * (i + 1) * tf.reduce_mean(p)
                    halt_loss += layer_cost
                    
                    # Track cumulative halting
                    cumulative_halt += tf.reduce_mean(p)
                
                # Encourage at least some computation to pass through 
                min_compute_penalty = tf.maximum(0.0, 0.2 - cumulative_halt)
                
                return task_loss + halt_loss + min_compute_penalty
            
            # Use our custom loss
            loss = adaptive_loss
        
        super(NeuraMoETransformer, self).compile(
            optimizer=optimizer, 
            loss=loss,
            metrics=metrics,
            **kwargs
        )

def create_neuramoe_transformer(input_shape, task_type='classification', num_classes=None, complexity=3):
    """Create NeuraMoE Transformer model with configurable complexity"""
    input_dim = input_shape[0]
    
    # Scale complexity parameters based on desired complexity
    expert_units = 32 * complexity
    kan_units = 24 * complexity
    num_experts = 3 + complexity // 2
    attn_heads = 2 + complexity // 2
    transformer_blocks = 1 + complexity // 2
    
    model = NeuraMoETransformer(
        input_dim=input_dim,
        num_classes=num_classes,
        num_experts=num_experts,
        expert_units=expert_units,
        attention_heads=attn_heads,
        kan_units=kan_units,
        transformer_blocks=transformer_blocks,
        dropout_rate=0.2,
        use_adaptive_computation=True,
        task_type=task_type
    )
    
    # Compile model with appropriate loss and metrics
    if task_type == 'classification':
        if num_classes is None or num_classes == 2:  # Binary classification
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:  # Multi-class classification
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
    else:  # Regression
        loss = 'mean_squared_error'
        metrics = ['mae']
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=metrics
    )
    
    return model

def train_and_evaluate_neuramoe_transformer(X_train, y_train, X_val, y_val, X_test, y_test, 
                                            task_type='classification', batch_size=32, epochs=50,
                                            complexity=3):
    """Train and evaluate a NeuraMoE Transformer model"""
    results = {}
    
    # Determine input shape and number of classes
    input_shape = (X_train.shape[1],)
    
    if task_type == 'classification':
        if len(y_train.shape) > 1:  # One-hot encoded
            num_classes = y_train.shape[1]
        else:
            num_classes = len(np.unique(y_train))
            # If binary classification, set to None
            if num_classes == 2:
                num_classes = None
    else:
        num_classes = None
    
    # Create a NeuraMoE Transformer model
    start_time = time.time()
    model = create_neuramoe_transformer(input_shape, task_type, num_classes, complexity)
    
    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Evaluate on test set
    inference_start_time = time.time()
    predictions = model.predict(X_test)
    inference_time = time.time() - inference_start_time
    
    # Calculate metrics based on task type
    if task_type == 'classification':
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Process predictions
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:  # Multi-class
            y_pred = np.argmax(predictions, axis=1)
            if len(y_test.shape) > 1:  # If y_test is one-hot encoded
                y_true = np.argmax(y_test, axis=1)
            else:
                y_true = y_test
        else:  # Binary
            y_pred = (predictions > 0.5).astype(int).flatten()
            y_true = y_test
        
        # Calculate metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, average='weighted')
        results['recall'] = recall_score(y_true, y_pred, average='weighted')
        results['f1'] = f1_score(y_true, y_pred, average='weighted')
    
    else:  # Regression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        y_pred = predictions.flatten()
        
        # Calculate metrics
        results['mae'] = mean_absolute_error(y_test, y_pred)
        results['mse'] = mean_squared_error(y_test, y_pred)
        results['rmse'] = np.sqrt(results['mse'])
        results['r2'] = r2_score(y_test, y_pred)
    
    # Store training history and time
    results['history'] = history.history
    results['training_time'] = training_time
    results['inference_time'] = inference_time
    results['inference_time_per_sample'] = inference_time / len(X_test)
    results['model_params'] = model.count_params()
    
    return results

def run_comprehensive_comparison(dataset_path, task_type='classification', batch_size=32, epochs=50):
    """
    Run a comprehensive comparison between Traditional ANN and NeuraMoE Transformer
    """
    # Import functions from the provided code
    from paste import prepare_data, create_traditional_ann, train_and_evaluate_model
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(dataset_path, task_type)
    
    # Get input shape
    input_shape = (X_train.shape[1],)
    
    # Get number of classes for classification
    if task_type == 'classification':
        if len(y_train.shape) > 1:  # One-hot encoded
            num_classes = y_train.shape[1]
        else:
            num_classes = len(np.unique(y_train))
            # If binary classification, set to None
            if num_classes == 2:
                num_classes = None
    else:
        num_classes = None
    
    # Create and train traditional ANN
    traditional_ann = create_traditional_ann(input_shape, task_type, num_classes)
    traditional_results = train_and_evaluate_model(
        traditional_ann, "Traditional ANN", 
        X_train, y_train, X_val, y_val, X_test, y_test,
        task_type, batch_size, epochs
    )
    
    # Train and evaluate NeuraMoE Transformer
    neuramoe_results = train_and_evaluate_neuramoe_transformer(
        X_train, y_train, X_val, y_val, X_test, y_test,
        task_type, batch_size, epochs
    )
    
    return {
        'Traditional ANN': traditional_results,
        'NeuraMoE Transformer': neuramoe_results,
        'task_type': task_type
    }

def visualize_results(results):
    """Visualize performance comparison between models"""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    task_type = results['task_type']
    
    # Create comparison table
    if task_type == 'classification':
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    else:  # Regression
        metrics = ['mae', 'mse', 'rmse', 'r2']
        metric_names = ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error', 'R² Score']
    
    # Prepare data for table
    comparison_data = []
    for metric, name in zip(metrics, metric_names):
        ann_value = results['Traditional ANN'][metric]
        neuramoe_value = results['NeuraMoE Transformer'][metric]
        
        # Calculate percentage improvement
        if metric in ['mae', 'mse', 'rmse']:  # Lower is better
            improvement = (ann_value - neuramoe_value) / ann_value * 100
            direction = '↓'
        else:  # Higher is better
            improvement = (neuramoe_value - ann_value) / ann_value * 100
            direction = '↑'
        
        comparison_data.append({
            'Metric': name,
            'Traditional ANN': f"{ann_value:.6f}",
            'NeuraMoE Transformer': f"{neuramoe_value:.6f}",
            'Improvement': f"{improvement:.2f}% {direction}"
        })
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print table
    print("\nPERFORMANCE METRICS COMPARISON:")
    print("-" * 60)
    print(comparison_df.to_string(index=False))
    
    # Model efficiency comparison
    ann_params = results['Traditional ANN']['model_params']
    neuramoe_params = results['NeuraMoE Transformer']['model_params']
    
    ann_inference = results['Traditional ANN']['inference_time_per_sample'] * 1000  # ms
    neuramoe_inference = results['NeuraMoE Transformer']['inference_time_per_sample'] * 1000  # ms
    
    print("\nMODEL EFFICIENCY COMPARISON:")
    print("-" * 60)
    print(f"Parameter Count - Traditional ANN: {ann_params:,}")
    print(f"Parameter Count - NeuraMoE Transformer: {neuramoe_params:,}")
    print(f"Parameters Ratio: {neuramoe_params/ann_params:.2f}x")
    print(f"Inference Time - Traditional ANN: {ann_inference:.4f} ms/sample")
    print(f"Inference Time - NeuraMoE Transformer: {neuramoe_inference:.4f} ms/sample")
    print(f"Inference Time Ratio: {neuramoe_inference/ann_inference:.2f}x")
    
    # Training convergence comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['Traditional ANN']['history']['loss'], label='ANN Training')
    plt.plot(results['Traditional ANN']['history']['val_loss'], label='ANN Validation')
    plt.plot(results['NeuraMoE Transformer']['history']['loss'], label='NeuraMoE Training')
    plt.plot(results['NeuraMoE Transformer']['history']['val_loss'], label='NeuraMoE Validation')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if task_type == 'classification':
        metric = 'accuracy'
    else:
        metric = 'mae'
    
    plt.plot(results['Traditional ANN']['history'][metric], label='ANN Training')
    plt.plot(results['Traditional ANN']['history']['val_' + metric], label='ANN Validation')
    plt.plot(results['NeuraMoE Transformer']['history'][metric], label='NeuraMoE Training')
    plt.plot(results['NeuraMoE Transformer']['history']['val_' + metric], label='NeuraMoE Validation')
    plt.title(f"{metric.upper()} Curves")
    plt.xlabel('Epoch')
    plt.ylabel(metric.upper())
    plt.legend()
    
    plt.tight_layout()
    plt.show()
# Import necessary libraries
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

# Load breast cancer dataset
cancer = load_breast_cancer()
df = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                columns=np.append(cancer['feature_names'], ['target']))
df.to_csv("breast_cancer_dataset.csv", index=False)

# Run comparison
results = run_comprehensive_comparison("breast_cancer_dataset.csv", 
                                    task_type='classification', 
                                    batch_size=32, 
                                    epochs=30)

# Visualize results
visualize_results(results)
