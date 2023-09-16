import tensorflow as tf
import numpy as np
import os
print("Tensorflow version={}".format(tf.__version__))

text = open('gabungan.txt','rb').read().decode(encoding="utf-8")
print(f"{len(text)} karakter kata")
idx = 8091
print(text[idx:idx+500])
text = text [8091:]
vocab = sorted(set(text))
print(f"{len(vocab)} token unik")
char2idx = {u:i for i,u in enumerate(vocab)}
print(char2idx)
indx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])
print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')
seq_length = 100
exsample_per_epoch = len(text)//(seq_length+1)
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
for i in char_dataset.take(10):
    print(indx2char[i.numpy()])
sequence = char_dataset.batch(seq_length+1 , drop_remainder=True)
def split_input_target(chunk):
    inputs = chunk[:-1]
    target = chunk[1:]
    return inputs , target
dataset = sequence.map(split_input_target)
BATCH_SIZE = 128
BUFFER_SIZE = 1000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE , drop_remainder=True)
def build_model(vocab_size , ebendding_dim , rnn_unit , batch_size , is_bidirectional=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size , ebendding_dim,
                                        batch_input_shape = [batch_size,None]))
    if is_bidirectional:
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(rnn_unit , return_sequences=True , 
                                 stateful = True,
                                 recurrent_initializer='glorot_uniform')
        ))
    else :
        model.add(tf.keras.layers.LSTM(
            rnn_unit , return_sequences=True , 
            stateful=True , recurrent_initializer='glorot_uniform'))
    model.add(tf.keras.layers.LSTM(rnn_unit,
                                   return_sequences=True,
                                   stateful=True,
                                   recurrent_initializer='glorot_uniform'))
    model.add(tf.keras.layers.Dense(vocab_size))
    return model
vocab_size = len(vocab)
embedding_dim = 256
rnn_unit = 1024
models = build_model(
    vocab_size=vocab_size,
    ebendding_dim=embedding_dim,
    rnn_unit = rnn_unit,
    batch_size= BATCH_SIZE
)
def loss(labels , logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels,logits,from_logits=True)
models.compile(optimizer="adam",loss=loss)
model_dir = r'data_baru/training_chekpoint'
chekpoint_prefix = os.path.join(model_dir , "ckpt_{epoch}")
chekpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=chekpoint_prefix,
    save_weights_only=True
)
epoch = 100
#history = models.fit(dataset , epochs=epoch , callbacks=[chekpoint])
model = build_model(vocab_size, embedding_dim, rnn_unit, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(model_dir))
print(tf.train.latest_checkpoint(model_dir))
model.build(tf.TensorShape([1, None]))
#model.summary()
def Generator_text(model , mode = "greedy" , context = "hello" , num_gen = 10 , temperature = 1.0):
    inputs_eval = [char2idx[c] for c in context]
    inputs_eval = tf.expand_dims(inputs_eval,0)
    text_gen = []
    model.reset_states()
    for i in range(num_gen):
        prediction = model(inputs_eval)
        prediction = tf.squeeze(prediction,0)
        if mode == "greedy":
            prediction_id = np.argmax(prediction[0])
        elif mode == "sampling":
            prediction = prediction / temperature
            prediction_id = tf.random.categorical(prediction , num_samples=1)[-1,0].numpy()
        inputs_eval = tf.expand_dims([prediction_id] , 0)
        text_gen.append(indx2char[prediction_id])
    return (context + " " + ''.join(text_gen))
#print("GREEDY")
#text_pair = Generator_text(model, mode= 'greedy', context="Assalamu'alaikum.",num_gen=700)
#print(text_pair)
#print("TOP_K")
#text_pair1 = Generator_text(model, mode= 'sampling', context="sayang ku",num_gen=700 , temperature=0.1)
#print(text_pair1)

