from base.base_model import BaseModel
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import layers

print(tf.__version__)

class BitcoinSentimentLSTM1Model(BaseModel):
    def __init__(self, optimizer, config):
        super(BitcoinSentimentLSTM1Model, self).__init__(optimizer, config)
        self.build_model()

    def build_model(self):

        top_words = 10000
        max_article_length = 100
        #embedding_vector_length = 60
        embedding_vector_length = self.config.trainer.embedding_vector_length
        #lstm_out = 15
        lstm_out = self.config.trainer.units

        self.model = Sequential()
        self.model.add(layers.Embedding(top_words, embedding_vector_length,
                input_length=max_article_length))
        # self.model.add(layers.Bidirectional(layers.LSTM(lstm_out, return_sequences=True)))
        self.model.add(layers.LSTM(lstm_out))
        # self.model.add(layers.LeakyReLU(alpha=0.5))
        # self.model.add(layers.Dropout(0.2))
        # self.model.add(layers.Flatten())
        # self.model.add(layers.Dense(10, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.summary()

        self.model.compile(loss='binary_crossentropy',
                    optimizer=self.optimizer, metrics=['accuracy'])

class BitcoinSentimentLSTM2Model(BaseModel):
    def __init__(self, optimizer, config):
        super(BitcoinSentimentLSTM2Model, self).__init__(optimizer, config)
        self.build_model()

    def build_model(self):

        top_words = 10000
        max_article_length = 100
        #embedding_vector_length = 60
        embedding_vector_length = self.config.trainer.embedding_vector_length
        #lstm_out = 15
        lstm_out = self.config.trainer.units

        self.model = Sequential()
        self.model.add(layers.Embedding(top_words, embedding_vector_length,     
                                     input_length=max_article_length) )
        self.model.add(layers.SpatialDropout1D(0.25))
        self.model.add(layers.LSTM(50, dropout=0.5, recurrent_dropout=0.5))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.summary()

        self.model.compile(loss='binary_crossentropy',
                    optimizer=self.optimizer, metrics=['accuracy'])

class BitcoinSentimentCNN1Model(BaseModel):
    def __init__(self, optimizer, config):
        super(BitcoinSentimentCNN1Model, self).__init__(optimizer, config)
        self.build_model()

    def build_model(self):

        top_words = 10000
        max_article_length = 100
        #embedding_vector_length = 60
        embedding_vector_length = self.config.trainer.embedding_vector_length
        
        self.model = Sequential()
        self.model.add(layers.Embedding(top_words, embedding_vector_length,
                input_length=max_article_length))

        # Convolutional model (3x conv, flatten, 2x dense)
        self.model.add(layers.Convolution1D(64, 3, padding='same'))
        self.model.add(layers.Convolution1D(32, 3, padding='same'))
        self.model.add(layers.Convolution1D(16, 3, padding='same'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(180, activation='sigmoid'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.summary()

        self.model.compile(loss='binary_crossentropy',
              optimizer=self.optimizer, metrics=['accuracy'])

class BitcoinSentimentCNN2Model(BaseModel):
    def __init__(self, optimizer, config):
        super(BitcoinSentimentCNN2Model, self).__init__(optimizer, config)
        self.build_model()

    def build_model(self):

        top_words = 10000
        max_article_length = 100
        # embedding_vector_length = 60
        #max_article_length = self.config.trainer.max_article_length
        embedding_vector_length = self.config.trainer.embedding_vector_length
        
        self.model = Sequential()
        self.model.add(layers.Embedding(top_words, embedding_vector_length,
                input_length=max_article_length))

        # Convolutional model (3x conv, flatten, 2x dense)
        self.model = Sequential()
        self.model.add(layers.Embedding(top_words, embedding_vector_length,
                input_length=max_article_length))
        self.model.add(layers.Conv1D(self.config.trainer.filters, self.config.trainer.kernal_size, activation='relu'))
        self.model.add(layers.MaxPooling1D(5))
        self.model.add(layers.Conv1D(self.config.trainer.filters, self.config.trainer.kernal_size, activation='relu'))
        self.model.add(layers.GlobalMaxPooling1D())
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.summary()

        self.model.compile(loss='binary_crossentropy',
              optimizer=self.optimizer, metrics=['accuracy'])

class BitcoinSentimentCNN3Model(BaseModel):
    def __init__(self, optimizer, config):
        super(BitcoinSentimentCNN3Model, self).__init__(optimizer, config)
        self.build_model()

    def build_model(self):

        top_words = 10000
        max_article_length = 100
        # embedding_vector_length = 60
        #max_article_length = self.config.trainer.max_article_length
        embedding_vector_length = self.config.trainer.embedding_vector_length

        # Convolutional model (3x conv, flatten, 2x dense) 100, 32, 8
        self.model = Sequential()
        self.model.add(layers.Embedding(top_words, embedding_vector_length, input_length=max_article_length))
        self.model.add(layers.Conv1D(self.config.trainer.filters, self.config.trainer.kernal_size, activation='relu'))
        self.model.add(layers.MaxPooling1D(pool_size=2))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(10, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.summary()

        self.model.compile(loss='binary_crossentropy',
              optimizer=self.optimizer, metrics=['accuracy'])

# class BitcoinSentimentTransformerModel(BaseModel):
#     def __init__(self, optimizer, config):
#         super(BitcoinSentimentTransformerModel, self).__init__(optimizer, config)
#         self.build_model()

#     def build_model(self):

#         top_words = 10000
#         max_article_length = 100
#         # embedding_vector_length = 60
#         #max_article_length = self.config.trainer.max_article_length
#         embedding_vector_length = self.config.trainer.embedding_vector_length

#         # Convolutional model (3x conv, flatten, 2x dense) 100, 32, 8
#         # self.model = Sequential()
#         # self.model.add(layers.Embedding(top_words, embedding_vector_length, input_length=max_article_length))
#         # self.model.add(layers.Conv1D(self.config.trainer.filters, self.config.trainer.kernal_size, activation='relu'))
#         # self.model.add(layers.MaxPooling1D(pool_size=2))
#         # self.model.add(layers.Flatten())
#         # self.model.add(layers.Dense(10, activation='relu'))
#         # self.model.add(layers.Dense(1, activation='sigmoid'))

#         embed_dim = 32  # Embedding size for each token
#         num_heads = 2  # Number of attention heads
#         ff_dim = 32  # Hidden layer size in feed forward network inside transformer

#         inputs = layers.Input(shape=(max_article_length,))
#         embedding_layer = TokenAndPositionEmbedding(max_article_length, top_words, embed_dim)
#         x = embedding_layer(inputs)
#         transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
#         x = transformer_block(x)
#         x = layers.GlobalAveragePooling1D()(x)
#         x = layers.Dropout(0.1)(x)
#         x = layers.Dense(20, activation="relu")(x)
#         x = layers.Dropout(0.1)(x)
#         outputs = layers.Dense(2, activation="softmax")(x)

#         self.model = keras.Model(inputs=inputs, outputs=outputs)
#         self.model.summary()

#         self.model.compile(loss='binary_crossentropy',
#               optimizer=self.optimizer, metrics=['accuracy'])

# class TransformerBlock(layers.Layer):
#     def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
#         super(TransformerBlock, self).__init__()
#         self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
#         self.ffn = keras.Sequential(
#             [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
#         )
#         self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
#         self.dropout1 = layers.Dropout(rate)
#         self.dropout2 = layers.Dropout(rate)

#     def call(self, inputs, training):
#         attn_output = self.att(inputs, inputs)
#         attn_output = self.dropout1(attn_output, training=training)
#         out1 = self.layernorm1(inputs + attn_output)
#         ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output, training=training)
#         return self.layernorm2(out1 + ffn_output)

# class TokenAndPositionEmbedding(layers.Layer):
#     def __init__(self, maxlen, vocab_size, embed_dim):
#         super(TokenAndPositionEmbedding, self).__init__()
#         self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
#         self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

#     def call(self, x):
#         maxlen = tf.shape(x)[-1]
#         positions = tf.range(start=0, limit=maxlen, delta=1)
#         positions = self.pos_emb(positions)
#         x = self.token_emb(x)
#         return x + positions

