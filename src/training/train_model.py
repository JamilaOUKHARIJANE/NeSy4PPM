
from __future__ import print_function, division

import itertools
import pdb
import copy
import os
import numpy as np
from hvplot import output
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.src.layers import LSTM, Dense, Input, Dropout, BatchNormalization, GlobalMaxPooling1D, Embedding, Concatenate
from keras.src.models import Model
from keras.src.optimizers import Nadam, Adam
#from keras.src.utils import plot_model
from keras_nlp.src.layers import TransformerEncoder, SinePositionEncoding
from tensorflow import distribute
import tensorflow as tf
from src.evaluation.prepare_data import prepare_encoded_data
from src.commons import shared_variables as shared
from src.commons.log_utils import LogData
from src.commons.utils import extract_trace_sequences
from src.training.Modulator import Modulator
from src.training.train_common import create_checkpoints_path, plot_loss


def _build_model(max_len, num_features, target_chars, target_chars_group, models_folder, resource, outcome):
    print('Build model...')
    if shared.One_hot_encoding:
        main_input = Input(shape=(max_len, num_features), name='main_input')
    elif shared.use_modulator:
        act_input = Input(shape=(num_features,), name='act_input')
        if resource:
            group_input = Input(shape=(num_features,), name='group_input')
            embedding_res = Embedding(
                input_dim=len(target_chars_group), output_dim=32)(group_input)
        embedding_act = Embedding(input_dim=len(target_chars), output_dim=32)(act_input)

        positional_encoding_act = SinePositionEncoding()(embedding_act)
        positional_encoding_res = SinePositionEncoding()(embedding_res)
        processed_act = embedding_act + positional_encoding_act
        processed_res = embedding_res + positional_encoding_res
    else:
        main_input = Input(shape=(num_features,), name='main_input')
        if resource:
            embedding = Embedding(input_dim=len(target_chars) * len(target_chars_group) if shared.combined_Act_res else len(target_chars) + len(target_chars_group)
                                  , output_dim=32)(main_input)
        else:
            embedding = Embedding(input_dim=len(target_chars), output_dim=32)(main_input)
        positional_encoding = SinePositionEncoding()(embedding)
        processed = embedding + positional_encoding

    if models_folder == "LSTM":
        if shared.One_hot_encoding:
            processed = LSTM(50, return_sequences=True, dropout=0.2)(main_input)
        elif shared.use_modulator:
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
        activity_output = Dense(len(target_chars), activation='softmax', name='act_output')(activity_output)

        if resource:
            if shared.use_modulator:
                processed = LSTM(50, return_sequences=True, dropout=0.2)(res_modulator)
            group_output = LSTM(50, return_sequences=False, dropout=0.2)(processed)
            group_output = BatchNormalization()(group_output)
            group_output = Dense(len(target_chars_group), activation='softmax', name='group_output')(group_output)

        if outcome:
            outcome_output = LSTM(50, return_sequences=False, dropout=0.2)(processed)
            outcome_output = BatchNormalization()(outcome_output)
            outcome_output = Dense(1, activation='sigmoid', name='outcome_output')(outcome_output)

        opt = Nadam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=3)  # schedule_decay=0.004,
    elif models_folder == "keras_trans":
        if shared.One_hot_encoding:
            processed = TransformerEncoder(intermediate_dim=64, num_heads=4)(main_input)
        elif shared.use_modulator:
            processed_act = TransformerEncoder(intermediate_dim=64, num_heads=4)(processed_act)
            processed_res = TransformerEncoder(intermediate_dim=64, num_heads=4)(processed_res)
            processed = Concatenate(axis=1)([processed_act, processed_res])
            act_modulator = Modulator(attr_idx=0, num_attrs=1, time=max_len)(processed)
            res_modulator = Modulator(attr_idx=1, num_attrs=1, time=max_len)(processed)
            processed = TransformerEncoder(intermediate_dim=64, num_heads=4)(act_modulator)
        else:
            processed = TransformerEncoder(intermediate_dim=64, num_heads=4)(processed)

        processed = GlobalMaxPooling1D()(processed)
        activity_output = Dense(len(target_chars), activation='softmax', name='act_output')(processed)

        if resource:
            if shared.use_modulator:
                processed = TransformerEncoder(intermediate_dim=64, num_heads=4)(res_modulator)
                processed = GlobalMaxPooling1D()(processed)
            group_output = Dense(len(target_chars_group), activation='softmax', name='group_output')(processed)

        if outcome:
            outcome_output = Dense(1, activation='sigmoid', name='outcome_output')(processed)

        opt = Adam()
    else:
        raise RuntimeError(f'The "{models_folder}" network is not defined!')

    if not resource and not outcome:
        model = Model(main_input, [activity_output])
        model.compile(loss={'act_output': 'categorical_crossentropy'}, optimizer=opt)

    elif resource and not outcome:
        if shared.use_modulator:
            model = Model(inputs=[act_input, group_input], outputs =[activity_output, group_output])
        else:
            model = Model(main_input, [activity_output, group_output])

        model.compile(loss={'act_output': 'categorical_crossentropy', 'group_output': 'categorical_crossentropy'},
                      optimizer=opt)

    elif resource and outcome:
        model = Model(main_input, [activity_output, group_output, outcome_output])
        model.compile(loss={'act_output': 'categorical_crossentropy', 'group_output': 'categorical_crossentropy',
                            'outcome_output': 'binary_crossentropy'}, optimizer=opt)
    #model.summary()
    models_folder += "_One_hot" * shared.One_hot_encoding + \
                     "_Combined_Act_res" * shared.combined_Act_res + \
                     "_Multi_Enc" * shared.use_modulator + \
                     "_Simple_categorical" * (not shared.One_hot_encoding and not shared.combined_Act_res and not shared.use_modulator)
    #plot_model(model, to_file=f'model_architecture_{models_folder}.png',show_shapes=True, show_layer_names=True)
    return model

def _train_model(model, checkpoint_name, x, y_a, y_o, y_g):
    model_checkpoint = ModelCheckpoint(checkpoint_name, save_best_only=True)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                   min_delta=0.0001, cooldown=0, min_lr=0)

    early_stopping = EarlyStopping(monitor='val_loss', patience=7)

    if (y_g is None) and (y_o is None) :
        history = model.fit(x, {'act_output': y_a }, validation_split=shared.validation_split,
                            batch_size=16, verbose=2, callbacks=[early_stopping, model_checkpoint, lr_reducer],
                            epochs=shared.epochs)
    elif (y_g is None) and (y_o is not None) :
        history = model.fit(x, {'act_output': y_a, 'outcome_output': y_o},
                            validation_split=shared.validation_split, verbose=2, batch_size=16,
                            callbacks=[early_stopping, model_checkpoint, lr_reducer], epochs=shared.epochs)
    elif (y_g is not None) and (y_o is not None) :
        history = model.fit(x, {'act_output': y_a, 'outcome_output': y_o, 'group_output': y_g},
                            validation_split=shared.validation_split, verbose=2, batch_size=16,
                            callbacks=[early_stopping, model_checkpoint, lr_reducer], epochs=shared.epochs)
    elif (y_g is not None) and (y_o is None):
        if shared.use_modulator:
            history = model.fit({'act_input':x["x_act"],'group_input': x["x_group"]},
                                {'act_output': y_a, 'group_output': y_g},
                                validation_split=shared.validation_split, verbose=2, batch_size=16,
                                callbacks=[early_stopping, model_checkpoint, lr_reducer], epochs=shared.epochs)
        else:
            history = model.fit(x, {'act_output': y_a, 'group_output': y_g},
                                validation_split=shared.validation_split, verbose=2, batch_size=16,
                                callbacks=[early_stopping, model_checkpoint, lr_reducer], epochs=shared.epochs)

    plot_loss(history, os.path.dirname(checkpoint_name))


def train(log_data: LogData, models_folder: str, resource: bool, outcome: bool):

    training_lines, training_lines_group, training_outcomes = extract_trace_sequences(log_data, log_data.training_trace_ids, resource, outcome)
    chars, chars_group, act_to_int, target_act_to_int, target_int_to_act, res_to_int, target_res_to_int, target_int_to_res \
        = prepare_encoded_data(log_data, resource)
    # Adding '!' to identify end of trace
    training_lines = [x + '!' for x in training_lines]

    if resource:
        training_lines_group = [x + '!' for x in training_lines_group]
        target_chars_group = copy.copy(chars_group)
        target_chars_group.append('!')
        print(f'Total groups: {len(chars_group)} - Target groups: {len(target_chars_group)}')
        print('\t', chars_group)
    else:
        target_chars_group = None
    maxlen = log_data.maxlen

    # Next lines here to get all possible characters for events and annotate them with numbers
    target_chars = copy.copy(chars)
    target_chars.append('!')

    print(f'Total chars: {len(chars)} - Target chars: {len(target_chars)}')
    print('\t', chars)

    step = 1
    softness = 0

    sentences = []
    sentences_group = []
    sentences_o = []
    next_chars = []
    next_chars_group = []

    if not resource and  not outcome:
        for line in training_lines:
            for i in range(0, len(line), step):
                if i == 0:
                    continue
                # We add iteratively, first symbol of the line, then two first, three...
                sentences.append(line[0: i])
                next_chars.append(line[i])

        print('Num. of training sequences:', len(sentences))

        print('Vectorization...')
        if shared.One_hot_encoding:
            num_features = len(chars) + 1
            x = np.zeros((len(sentences),maxlen, num_features), dtype=np.float32)
        else:
            num_features = maxlen
            x = np.zeros((len(sentences), num_features), dtype=np.float32)

        print(f'Num. of features: {num_features}')
        y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
        y_g = None
        y_o = None
        for i, sentence in enumerate(sentences):
            leftpad = maxlen - len(sentence)
            for t, char in enumerate(sentence):
                if shared.One_hot_encoding:
                    for c in chars:
                        if c == char:
                            x[i, t + leftpad, act_to_int[c] - 1] = 1
                   # x[i, t + leftpad, len(chars)] = t + 1
                else:
                    x[i, t] = act_to_int[char]
            for c in target_chars:
                if c == next_chars[i]:
                    y_a[i, target_act_to_int[c] - 1] = 1 - softness
                else:
                    y_a[i, target_act_to_int[c] - 1] = softness / (len(target_chars) - 1)

        for fold in range(shared.folds):

            model = _build_model(maxlen, num_features, target_chars, target_chars_group, models_folder, resource, outcome)
            checkpoint_name = create_checkpoints_path(log_data.log_name.value, models_folder, fold, 'CF')
            _train_model(model, checkpoint_name, x, y_a, y_o, y_g)

    if resource and not outcome: #NON ENTRA QUI
        # Ensure both training_lines and training_lines_group have the same length
        if len(training_lines) != len(training_lines_group):
            raise ValueError("Mismatch in length of training_lines and training_lines_group")

        sentences = []
        sentences_group = []
        next_chars = []
        next_chars_group = []

        for line, line_group in zip(training_lines, training_lines_group):
            if len(line) != len(line_group):
                raise ValueError("Mismatch in length of line and line_group")
                
            for i in range(1, len(line)):
                # We add iteratively, first symbol of the line, then two first, three...
                sentences.append(line[0: i])
                sentences_group.append(line_group[0: i])

                next_chars.append(line[i])
                next_chars_group.append(line_group[i])

        print('Num. of training sequences:', len(sentences))
        print('Vectorization...')
        if shared.One_hot_encoding:
            num_features = len(chars) + len(chars_group) #+ 1
            x = np.zeros((len(sentences),maxlen, num_features), dtype=np.float32)
        else:
            if shared.combined_Act_res:
                result_list = [x + y for x, y in itertools.product(chars, chars_group)]
                target_to_int = dict((c, i + 1) for i, c in enumerate(result_list))
                num_features = maxlen
            else:
                num_features = maxlen * 2
            if shared.use_modulator:
                num_features = maxlen
                x_a = np.zeros((len(sentences), num_features), dtype=np.float32)
                x_g = np.zeros((len(sentences), num_features), dtype=np.float32)
                x = {
                    "x_act": x_a,
                    "x_group": x_g
                }
            else:
                x = np.zeros((len(sentences), num_features), dtype=np.float32)
        print(f'Num. of features: {num_features}')

        y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
        y_g = np.zeros((len(sentences), len(target_chars_group)), dtype=np.float32)
        y_o = None

        for i, sentence in enumerate(sentences):
            leftpad = maxlen - len(sentence)
            counter_act = 0
            counter_res = 1
            sentence_group = sentences_group[i]
            for t, char in enumerate(sentence):
                if shared.One_hot_encoding:
                    if char in chars:
                        x[i, t + leftpad, act_to_int[char] -1] = 1
                    if t < len(sentence_group) and sentence_group[t] in chars_group:
                        x[i, t + leftpad, len(chars) + res_to_int[sentence_group[t]] - 1] = 1
                    #x[i, t + leftpad, num_features - 1] = t + 1
                elif shared.use_modulator:
                    x_a[i, t] = act_to_int[char]
                    x_g[i,t] = res_to_int[sentence_group[t]]
                else:
                    if shared.combined_Act_res:
                        x[i, t] = target_to_int[char + sentence_group[t]]
                    else:
                        x[i, counter_act] = act_to_int[char]
                        x[i, counter_res] = res_to_int[sentence_group[t]]
                        counter_act += 2
                        counter_res += 2

            for c in target_chars:
                if c == next_chars[i]:
                    y_a[i, target_act_to_int[c]- 1] = 1 - softness
                else:
                    y_a[i, target_act_to_int[c]- 1] = softness / (len(target_chars) - 1)
            for c in target_chars_group:
                if c == next_chars_group[i]:
                    y_g[i, target_res_to_int[c]- 1] = 1 - softness
                else:
                    y_g[i, target_res_to_int[c]- 1] = softness / (len(target_chars_group) - 1)
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            for fold in range(shared.folds):
                model = _build_model(maxlen, num_features, target_chars, target_chars_group, models_folder, resource, outcome)
                checkpoint_name = create_checkpoints_path(log_data.log_name.value, models_folder, fold, 'CFR')
                _train_model(model, checkpoint_name, x, y_a, y_o, y_g)
    if ~resource and outcome:
        for line, outcome in zip(training_lines, training_outcomes):
            for i in range(0, len(line), step):
                if i == 0:
                    continue
                # We add iteratively, first symbol of the line, then two first, three...
                sentences.append(line[0: i])
                sentences_o.append(outcome)

                next_chars.append(line[i])

        print('Num. of training sequences:', len(sentences))
        print('Vectorization...')

        if shared.One_hot_encoding:
            num_features = len(chars) + 1
            x = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
        else:
            num_features = maxlen
            x = np.zeros((len(sentences), num_features), dtype=np.float32)
        print(f'Num. of features: {num_features}')

        y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
        y_g = None
        y_o = np.zeros((len(sentences)), dtype=np.float32)

        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                if shared.One_hot_encoding:
                    for c in chars:
                        if c == char:
                            x[i, t + leftpad, act_to_int[c] -1 ] = 1
                    x[i, t + leftpad, len(chars)] = t + 1
                else:
                    x[i, t] = act_to_int[char]

            for c in target_chars:
                if c == next_chars[i]:
                    y_a[i, target_act_to_int[c]- 1] = 1 - softness
                else:
                    y_a[i, target_act_to_int[c]- 1] = softness / (len(target_chars) - 1)
            y_o[i] = sentences_o[i]

        for fold in range(shared.folds):
            model = _build_model(maxlen, num_features, target_chars, target_chars_group, models_folder, resource, outcome)
            checkpoint_name = create_checkpoints_path(log_data.log_name.value, models_folder, fold, 'CFO')
            _train_model(model, checkpoint_name, x, y_a, y_o, y_g)
    if resource and outcome:
        for line, line_group, outcome in zip(training_lines, training_lines_group, training_outcomes):
            for i in range(0, len(line), step):
                if i == 0:
                    continue
                # We add iteratively, first symbol of the line, then two first, three...
                sentences.append(line[0: i])
                sentences_group.append(line_group[0: i])
                sentences_o.append(outcome)

                next_chars.append(line[i])
                next_chars_group.append(line_group[i])

        print('Num. of training sequences:', len(sentences))
        print('Vectorization...')
        if shared.One_hot_encoding:
            num_features = len(chars) + len(chars_group) + 1
            x = np.zeros((len(sentences),maxlen, num_features), dtype=np.float32)
        else:
            if shared.combined_Act_res:
                result_list = [x + y for x, y in itertools.product(chars, chars_group)]
                target_to_int = dict((c, i + 1) for i, c in enumerate(result_list))
                num_features = maxlen
            else:
                num_features = maxlen * 2
            x = np.zeros((len(sentences), num_features), dtype=np.float32)
        print(f'Num. of features: {num_features}')

        y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
        y_g = np.zeros((len(sentences), len(target_chars_group)), dtype=np.float32)
        y_o = np.zeros((len(sentences)), dtype=np.float32)

        result_list = [x + y for x, y in itertools.product(chars, chars_group)]
        target_to_int = dict((c, i + 1) for i, c in enumerate(result_list))
        for i, sentence in enumerate(sentences):
            leftpad = maxlen - len(sentence)
            counter_act = 0
            counter_res = 1
            sentence_group = sentences_group[i]
            for t, char in enumerate(sentence):
                if shared.One_hot_encoding:
                    if char in chars:
                        x[i, t + leftpad, act_to_int[char] - 1] = 1
                    if t < len(sentence_group) and sentence_group[t] in chars_group:
                        x[i, t + leftpad, len(chars) + res_to_int[sentence_group[t]] - 1] = 1
                    x[i, t + leftpad, num_features - 1] = t + 1
                else:
                    if shared.combined_Act_res:
                        x[i, t] = target_to_int[char + sentence_group[t]]
                    else:
                        x[i, counter_act] = act_to_int[char]
                        x[i, counter_res] = res_to_int[sentence_group[t]]
                        counter_act += 2
                        counter_res += 2

            for c in target_chars:
                if c == next_chars[i]:
                    y_a[i, target_act_to_int[c]- 1] = 1 - softness
                else:
                    y_a[i, target_act_to_int[c]- 1] = softness / (len(target_chars) - 1)
            for c in target_chars_group:
                if c == next_chars_group[i]:
                    y_g[i, target_res_to_int[c]- 1] = 1 - softness
                else:
                    y_g[i, target_res_to_int[c] - 1] = softness / (len(target_chars_group) - 1)
            y_o[i] = sentences_o[i]

        for fold in range(shared.folds):
            model = _build_model(maxlen, num_features, target_chars, target_chars_group, models_folder, resource, outcome)
            checkpoint_name = create_checkpoints_path(log_data.log_name.value, models_folder, fold, 'CFRO')
            _train_model(model, checkpoint_name, x, y_a, y_o, y_g)

