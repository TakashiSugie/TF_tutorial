#初心者のためのTensorFlow2.0 
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'#これで警告とかが出ない


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train[:1].shape)
#sequential
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
#未学習モデルに1,28,28の訓練データを一枚入れた時にロジットを求め、それをnumpyに
#ちなみにロジットとは0~1の確率を-無限から無限にしたやつ、modelに画像を入れるとそうなるらしい
predictions = model(x_train[:1]).numpy()
predictions
#ロジットを確率に変換
tf.nn.softmax(predictions).numpy()
#サンプルに対してクラスごとの損失を返すやつを設定
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#訓練されていないモデルにサンプルを入れ、その時の損失を返すやつ
#（正しいクラスに対して、どれくらい高く予測ができるか）完全に正しければ確率1となるので、損失は0になる
loss_fn(y_train[:1], predictions).numpy()
print(loss_fn(y_train[:1], predictions).numpy())
#モデルをコンパイル
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
#これはfitを用いた訓練
#model.fit(x_train, y_train, epochs=5)
#これでモデルの評価ができる
model.evaluate(x_test,y_test,verbose = 2)
#各クラスに対して確率を求めたいのであればこんな感じ
#もともとのモデルをラップしてやって機能追加
#modelをもとに、Probablity_modelという新しい確率を返すモデルを作成
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
print(x_test[:5,:,:].shape)
print(x_test[:5].shape)
#print(probability_model(x_test[:5]))
