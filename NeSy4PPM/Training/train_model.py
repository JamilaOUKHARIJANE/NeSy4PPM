
from __future__ import print_function, division

import os
from pathlib import Path

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM, Dense, Input, BatchNormalization, GlobalMaxPooling1D, Embedding, Concatenate
from keras.models import Model
from keras.optimizers import Nadam, Adam
from keras_nlp.layers import TransformerEncoder, SinePositionEncoding
from NeSy4PPM.Data_preprocessing import shared_variables as shared
from NeSy4PPM.Data_preprocessing.log_utils import LogData
from NeSy4PPM.Data_preprocessing.utils import Encodings, NN_model
from NeSy4PPM.Training.Modulator import Modulator
from NeSy4PPM.Data_preprocessing.data_preprocessing import extract_encode_prefixes
from NeSy4PPM.Training.train_common import create_checkpoints_path, plot_loss
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


def _build_model(max_len, num_features, target_chars, target_chars_group, model_arch:NN_model, resource, encoder):
    print('Build model...')
    if encoder == Encodings.One_hot:
        main_input = Input(shape=(max_len, num_features), name='main_input')
    elif encoder == Encodings.Multi_encoders:
        act_input = Input(shape=(num_features,), name='act_input')
        if resource:
            group_input = Input(shape=(num_features,), name='group_input')
            embedding_res = Embedding(
                input_dim=target_chars_group, output_dim=32)(group_input)
        embedding_act = Embedding(input_dim=target_chars, output_dim=32)(act_input)

        positional_encoding_act = SinePositionEncoding()(embedding_act)
        positional_encoding_res = SinePositionEncoding()(embedding_res)
        processed_act = embedding_act + positional_encoding_act
        processed_res = embedding_res + positional_encoding_res
    else:
        main_input = Input(shape=(num_features,), name='main_input')
        if resource:
            embedding = Embedding(input_dim=target_chars * target_chars_group if encoder==Encodings.Shrinked_based else target_chars + target_chars_group
                                  , output_dim=32)(main_input)
        else:
            embedding = Embedding(input_dim=target_chars, output_dim=32)(main_input)
        positional_encoding = SinePositionEncoding()(embedding)
        processed = embedding + positional_encoding

    if model_arch == NN_model.LSTM:
        if encoder == Encodings.One_hot:
            processed = LSTM(50, return_sequences=True, dropout=0.2)(main_input)
        elif encoder == Encodings.Multi_encoders:
            processed_act = LSTM(50, return_sequences=True, dropout=0.2)(processed_act)
            processed_res = LSTM(50, return_sequences=True, dropout=0.2)(processed_res)
            processed = Concatenate(axis=1)([processed_act, processed_res])
            processed = BatchNormalization()(processed)
            act_modulator = Modulator(attr_idx=0, num_attrs=1, time=max_len)(processed)
            res_modulator = Modulator(attr_idx=1, num_attrs=1, time=max_len)(processed)
            processed = LSTM(50, return_sequences=True, dropout=0.2)(act_modulator)
        else:
            processed = LSTM(50, return_sequences=True, dropout=0.2)(processed)
            processed = BatchNormalization()(processed)
        activity_output = LSTM(50, return_sequences=False, dropout=0.2)(processed)
        activity_output = BatchNormalization()(activity_output)
        activity_output = Dense(target_chars, activation='softmax', name='act_output')(activity_output)

        if resource:
            if encoder == Encodings.Multi_encoders:
                processed = LSTM(50, return_sequences=True, dropout=0.2)(res_modulator)
            group_output = LSTM(50, return_sequences=False, dropout=0.2)(processed)
            group_output = BatchNormalization()(group_output)
            group_output = Dense(target_chars_group, activation='softmax', name='group_output')(group_output)

        opt = Nadam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=3)  # schedule_decay=0.004,
    elif model_arch == NN_model.Transformer:
        if encoder == Encodings.One_hot:
            processed = TransformerEncoder(intermediate_dim=64, num_heads=4)(main_input)
        elif encoder == Encodings.Multi_encoders:
            processed_act = TransformerEncoder(intermediate_dim=64, num_heads=4)(processed_act)
            processed_res = TransformerEncoder(intermediate_dim=64, num_heads=4)(processed_res)
            processed = Concatenate(axis=1)([processed_act, processed_res])
            act_modulator = Modulator(attr_idx=0, num_attrs=1, time=max_len)(processed)
            res_modulator = Modulator(attr_idx=1, num_attrs=1, time=max_len)(processed)
            processed = TransformerEncoder(intermediate_dim=64, num_heads=4)(act_modulator)
        else:
            processed = TransformerEncoder(intermediate_dim=64, num_heads=4)(processed)

        processed = GlobalMaxPooling1D()(processed)
        activity_output = Dense(target_chars, activation='softmax', name='act_output')(processed)

        if resource:
            if encoder == Encodings.Multi_encoders:
                processed = TransformerEncoder(intermediate_dim=64, num_heads=4)(res_modulator)
                processed = GlobalMaxPooling1D()(processed)
            group_output = Dense(target_chars_group, activation='softmax', name='group_output')(processed)

        opt = Adam()
    else:
        raise RuntimeError(f'The "{model_arch.value}" network is not defined!')

    if resource:
        if encoder == Encodings.Multi_encoders:
            model = Model(inputs=[act_input, group_input], outputs =[activity_output, group_output])
        else:
            model = Model(main_input, [activity_output, group_output])

        model.compile(loss={'act_output': 'categorical_crossentropy', 'group_output': 'categorical_crossentropy'},
                      optimizer=opt)
    else:
        model = Model(main_input, [activity_output])
        model.compile(loss={'act_output': 'categorical_crossentropy'}, optimizer=opt)

    #plot_model(model, to_file=f'model_architecture_{models_folder}.png',show_shapes=True, show_layer_names=True)
    return model

def _train_model(model, checkpoint_name, x, y_a, y_g,encoder):
    model_checkpoint = ModelCheckpoint(checkpoint_name, save_best_only=True)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                   min_delta=0.0001, cooldown=0, min_lr=0)

    early_stopping = EarlyStopping(monitor='val_loss', patience=7)

    if (y_g is None) :
        history = model.fit(x, {'act_output': y_a }, validation_split=shared.validation_split,
                            batch_size=16, verbose=2, callbacks=[early_stopping, model_checkpoint, lr_reducer],
                            epochs=shared.epochs)
    else:
        if encoder == Encodings.Multi_encoders:
            history = model.fit({'act_input':x["x_act"],'group_input': x["x_group"]},
                                {'act_output': y_a, 'group_output': y_g},
                                validation_split=shared.validation_split, verbose=2, batch_size=16,
                                callbacks=[early_stopping, model_checkpoint, lr_reducer], epochs=shared.epochs)
        else:
            history = model.fit(x, {'act_output': y_a, 'group_output': y_g},
                                validation_split=shared.validation_split, verbose=2, batch_size=16,
                                callbacks=[early_stopping, model_checkpoint, lr_reducer], epochs=shared.epochs)

    plot_loss(history, os.path.dirname(checkpoint_name))


def train(log_data: LogData, encoder:Encodings, model_arch: NN_model, X, y_a, y_g=None, output_folder:Path=shared.output_folder):
    maxlen = log_data.max_len
    if y_g is None:
        model_type='CF'
        resource=False
    else:
        model_type = 'CFR'
        resource = True
    model = _build_model(maxlen, X["x_act"].shape[-1] if encoder==Encodings.Multi_encoders else X.shape[-1], y_a.shape[1], y_g.shape[1] if y_g is not None else 0, model_arch, resource,encoder)
    checkpoint_name = create_checkpoints_path(log_data.log_name, model_arch,model_type, encoder, output_folder)
    _train_model(model, checkpoint_name, X, y_a, y_g,encoder)

def learn(log_data: LogData, encoder:Encodings, model_arch: NN_model, resource: bool,output_folder:Path=shared.output_folder):
    x, y_a, y_g = extract_encode_prefixes(log_data, encoder, resource)
    train(log_data, encoder=encoder, model_arch=model_arch, X=x, y_a=y_a, y_g=y_g, output_folder=output_folder)