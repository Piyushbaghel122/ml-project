import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

real_scores = [90, 60, 80, 100]
predicted_scores = [85, 70, 70, 95]

mae = mean_absolute_error(real_scores, predicted_scores)
mse = mean_squared_error(real_scores, predicted_scores)
rmse = np.sqrt(mse)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
