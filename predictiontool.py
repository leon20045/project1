import joblib
import numpy as np

My_Prediction = joblib.load('Projectleon.joblib')

Test = np.array([[9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3]])

Final_Prediction = My_Prediction.predict(Test)
print("Projectleon",Final_Prediction)