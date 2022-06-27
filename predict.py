from tensorflow.keras.models import load_model
from data_preprocess import preprocess_for_predict
model=load_model("mymodel.h5")

x = preprocess_for_predict([" abc s good morning america ranks it the NUMBER christmas toy of the season the new NUMBER inch mini remote control cars are out of stock everywhere parents are searching frantically but having no luck there are millions of kids expecting these for the holiday season lets hope somebody gets them in or santa may be in trouble dianne sawyer nov NUMBER sold out in all stores accross the country retail price is NUMBER NUMBER we have limited stock and free shipping for only NUMBER NUMBER hyperlink check out this years hottest toy hyperlink unsubscribe forever "])


output = model.predict(x)
if output>=0.5:
    print('spam')
else:
    print('Not spam')

