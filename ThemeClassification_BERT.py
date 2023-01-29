import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
df = pd .read_csv("Complete_Dataset Dr Muneera .csv")


bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

# 2 ways to design model sequential and functional model(one layer output give input to another layer )(input give same way as function give input)
#we use functional approach to design model
#First Declare Bert layer
text_input=tf.keras.layer.Input(shape=(),dtype=tf.string,name="text")
preprocessed_text=bert_preprocess(text_input)
outputs=bert_encoder(preprocessed_text)

#Neural network layer
l=tf.kearas.layers.Dropout(0.1,name="dropout")(outputs["pooled_output"])
l=tf.keras.layers.Dense(1,activation='sigmoid',name="output")(l)

#Construcat a final model
model=tf.keras .Model(inputs=[text_input],outputs=[l])
model.summary()
model.compile(optimizer="adam",loss="binary_crossentropy")
model.fit(X_train,Y_train,epochs=10)
y_predicted=model.predict(X_test)
y_predicted.flatten()
y_predicted=np.where(y_predicted > 0.5 ,1,0)
y_predicted 


