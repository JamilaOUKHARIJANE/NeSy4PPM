from pathlib import Path
from matplotlib import pyplot as plt
from keras import layers, Sequential

from NeSy4PPM.Data_preprocessing.utils import Encodings, NN_model


def create_checkpoints_path(log_name, model:NN_model, model_type,encoder:Encodings,output_folder):
    """
        Create a directory path and filename pattern for model checkpoints.

        Args:
            log_name (str): Name of the log.
            model (NN_model): Enum representing the model type.
            model_type (str): Specific model type (e.g., 'LSTM', 'Transformer').
            encoder (Encodings): Enum representing the encoding method.
            output_folder (Path): Base output folder path.

        Returns:
            str: Full path pattern for saving checkpoint files.
        """
    models_folder= model.value+'_'+ encoder.value
    folder_path = output_folder / models_folder / 'models' / model_type / log_name
    if not Path.exists(folder_path):
        Path.mkdir(folder_path, parents=True)
    checkpoint_name = folder_path / 'model_{epoch:03d}-{val_loss:.3f}.keras'
    #debugging checkpoint path
    print(f"Checkpoint path: {folder_path}")
    print(f"Checkpoint name pattern: {checkpoint_name}")
    return str(checkpoint_name)


def plot_loss(history, dir_name):
    """
        Plot and save the training and validation loss curves.

        Args:
            history (keras.callbacks.History): Keras training history object.
            dir_name (str or Path): Directory to save the plot image.
    """
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(str(Path(dir_name) / "loss.png"))


class CustomTransformer(layers.Layer):
    """
        Custom Transformer block consisting of multi-head attention and feed-forward layers.
    """
    def __init__(self, embed_dim=256, dense_dim=2048, num_heads=8, **kwargs):
        """
               Initialize the CustomTransformer layer.

               Args:
                   embed_dim (int): Embedding dimensionality.
                   dense_dim (int): Dimensionality of the dense projection layer.
                   num_heads (int): Number of attention heads.
        """
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None, *args, **kwargs):
        """
               Forward pass of the CustomTransformer layer.

               Args:
                   inputs (tf.Tensor): Input tensor.
                   mask (tf.Tensor, optional): Attention mask.

               Returns:
                   tf.Tensor: Output tensor after attention and feed-forward layers.
        """
        attention_output = self.attention(inputs, inputs)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        """
                Return the config dictionary for recreating this layer.

                Returns:
                    dict: Configuration parameters.
        """
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config
