import numpy as np
import tensorflow as tf

from keras.layers import BatchNormalization, ConvLSTM2D, Dense, Flatten, Input, MaxPooling3D, TimeDistributed
from keras.models import Model
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


class GenerativeModel:

    def __init__(self, seed_roll_length: int, notes_range: int):
        """
        :param seed_roll_length: sequence length - num of piano roll ticks to start with
        :param notes_range: length of range of supported notes
        """
        self.model = self._convlstm_model(seed_roll_length, notes_range)

    def _convlstm_model(self, seed_roll_length: int, notes_range: int) -> Model:
        """
        Construct model architecture
        """
        # Don't have channels
        model_input = Input(shape=(None, 1, seed_roll_length, notes_range), name='model_input')

        x = ConvLSTM2D(filters=64, kernel_size=(7, 7),
                       data_format='channels_first', activation='relu', padding='same',
                       dropout=0.1, return_sequences=True)(model_input)
        x = BatchNormalization()(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_first')(x)

        x = ConvLSTM2D(filters=32, kernel_size=(5, 5),
                       data_format='channels_first', activation='relu', padding='same',
                       recurrent_dropout=0.1, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D(pool_size=(1, 3, 3), padding='same', data_format='channels_first')(x)

        x = ConvLSTM2D(filters=16, kernel_size=(3, 3),
                       data_format='channels_first', activation='relu', padding='same',
                       recurrent_dropout=0.1, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_first')(x)

        x = TimeDistributed(Flatten())(x)
        model_output = TimeDistributed(Dense(notes_range))(x)

        model = Model(inputs=model_input, outputs=model_output, name='music-gen')

        model.compile(loss=BinaryCrossentropy(from_logits=True), optimizer='adam')

        return model

    def train(self, x_train, y_train, x_val, y_val, epochs=100, batch_size=10, early_stopping_patience=20, reduce_lr_patience=10):
        """
        Fit generative model
        """
        early_stopping = EarlyStopping(monitor="val_loss", patience=early_stopping_patience)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=reduce_lr_patience)

        self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping, reduce_lr],
        )

    def predict(self, seed_piece, num_ticks_to_generate: int, temperature: float = 1.0):
        """
        Predict continuation of the piece
        :param seed_piece: piece to start with (set tone?)
        :param num_ticks_to_generate: length of output piece
        :param temperature: control randomness of generated music
        :return: generated piece
        """
        # shape: [num pieces in seq, channel, roll length, num notes]
        pieces = np.expand_dims(seed_piece, axis=(0, 1))
        whole_generated_piece = []
        for _ in range(num_ticks_to_generate):
            pred = self.model.predict(np.expand_dims(pieces, axis=0))
            pred_logits = np.squeeze(pred, axis=0)[-1, :] / temperature
            predicted_chord = np.array([int(p > np.random.random()) for p in tf.sigmoid(pred_logits)])

            whole_generated_piece.append(predicted_chord)
            last_piece_shifted = pieces[-1, :, 1:, :]
            new_piece = np.hstack((last_piece_shifted, np.expand_dims(predicted_chord, axis=(0, 1))))
            pieces = np.vstack((pieces, np.expand_dims(new_piece, axis=0)))

        return np.vstack(whole_generated_piece)
