import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'#これで警告とかが出ない

optimizer = tf.optimizers.Adam(1.0e-4)
train_loss = tf.keras.metrics.Mean() # コスト記録用
train_acc = tf.keras.metrics.SparseCategoricalAccuracy() # 精度計算・記録用

@tf.function
def train_step(inputs):
    images, labels = inputs

    # tf.GtadientTapeブロックで入力からロスまで計算
    with tf.GradientTape() as tape:
        logits = model(images)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

    # gradientを計算
    # gradientを計算する重みをsourcesとして指定することが必須
    # keras.Modelを使っているとmodel.trainable_variablesで渡すことができて便利
    grad = tape.gradient(loss, sources=model.trainable_variables)

    # optimizerで重みを更新
    optimizer.apply_gradients(zip(grad, model.trainable_variables))

    # lossの値を記録
    train_loss.update_state(loss)
    # train_loss(loss) # このように単純に__call__しても良い

    # 精度を記録
    train_acc.update_state(labels, logits)
    # train_acc(labels, logits) # このように単純に__call__しても良い