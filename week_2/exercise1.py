import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Generate synthetic data
x1 = np.arange(0, 10, 0.1)
x2 = np.arange(0, 10, 0.1)
x1, x2 = np.meshgrid(x1, x2)
y = np.sin(x1) * np.cos(x2) + np.random.normal(scale=0.1, size=x1.shape)

# Flatten the arrays
x1 = x1.flatten()
x2 = x2.flatten()
y = y.flatten()
X = np.vstack((x1, x2)).T

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree regressor, where we take the max depth fomr 1 to 20, aswell as the splitter as best and random. return the r2 values and MSE in an array with both inputs, which will then be used to plot the data in two graphs, one for MSE and one for r2
def train_decision_tree(X_train, y_train, X_test, y_test):
    r2_values = []
    mse_values = []
    for splitter in ['best', 'random']:
        for i in range(1, 21):
            dt = DecisionTreeRegressor(max_depth=i, splitter=splitter)
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2_values.append(r2)
            mse_values.append(mse)
    return r2_values, mse_values

r2_values, mse_values = train_decision_tree(X_train, y_train, X_test, y_test)

# Plot the data in the same window
plt.figure(1)
plt.xticks(range(1, 21))
plt.plot(range(1, 21), r2_values[:20], label='Best Splitter')
plt.plot(range(1, 21), r2_values[20:], label='Random Splitter')
plt.xlabel('Max Depth')
plt.ylabel('R2 Score')
plt.title('R2 Score vs Max Depth')
plt.legend()

plt.figure(2)
plt.xticks(range(1, 21))
plt.plot(range(1, 21), mse_values[:20], label='Best Splitter')
plt.plot(range(1, 21), mse_values[20:], label='Random Splitter')
plt.xlabel('Max Depth')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs Max Depth')
plt.legend()
plt.show()




        












