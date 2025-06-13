from pathlib import Path
from matplotlib import pyplot as plt
from keras import layers, Sequential

from src.commons import shared_variables as shared


def create_checkpoints_path(log_name, models_folder, fold, model_type):
    models_folder += "_One_hot" * (shared.One_hot_encoding and not shared.use_modulator)+ \
                     "_Combined_Act_res" * (shared.combined_Act_res and not shared.use_modulator) + \
                     "_Multi_Enc" * (shared.use_modulator and not shared.One_hot_encoding) + \
                     "_Multi_One_hot_Enc" * (shared.use_modulator and shared.One_hot_encoding) + \
                     "_Simple_categorical" * (not shared.One_hot_encoding and not shared.combined_Act_res and not shared.use_modulator)
    folder_path = shared.output_folder / models_folder / str(fold) / 'models' / model_type / log_name
    if not Path.exists(folder_path):
        Path.mkdir(folder_path, parents=True)
    checkpoint_name = folder_path / 'model_{epoch:03d}-{val_loss:.3f}.keras'
    #debugging checkpoint path
    print(f"Checkpoint path: {folder_path}")
    print(f"Checkpoint name pattern: {checkpoint_name}")
    return str(checkpoint_name)


def plot_loss(history, dir_name):
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(str(Path(dir_name) / "loss.png"))


class CustomTransformer(layers.Layer):
    def __init__(self, embed_dim=256, dense_dim=2048, num_heads=8, **kwargs):
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
        attention_output = self.attention(inputs, inputs)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config
