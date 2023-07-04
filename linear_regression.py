import matplotlib.pyplot as plt, numpy as np, pandas as pd, mpl_toolkits.mplot3d
from sklearn.model_selection import train_test_split # splits data into training and test data
from sklearn.pipeline import make_pipeline # makes a pipeline
from sklearn.preprocessing import StandardScaler # scales data to standard normal distribution
from sklearn.decomposition import PCA # principal component analysis
from sklearn.linear_model import LinearRegression # linear regression

# Import data as a Pandas DataFrame and preprocess them for scikit-learn:
heating_data = pd.read_csv('Heating-data.csv', delimiter='\t', index_col='Date') # loads CSV to Pandas

features = ["Sunshine duration [h/day]", "Outdoor temperature [°C]", "Solar yield [kWh/day]", "Solar pump [h/day]", "Valve [h/day]"] # dependent variable

# Separate the target variable 'y' (gas consumption) from the other variables 'x'
target = "Gas consumption [kWh/day]" #dependent variable

x = np.c_[heating_data[features]] # extracts feature values as a matrix
y = np.c_[heating_data[target]] # extracts target values as a one-column matrix

# Choose by random 30 % of data as test data, i.e., 70 % as training data:
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

# Fit and predict with a pipeline of scaling, PCA, and linear regression:
pipe = make_pipeline(StandardScaler(), PCA(2), LinearRegression())
pipe.fit(X_train, y_train)

# Print model score:
print("score (train values): ", f"{pipe.score(X_train, y_train):.2%}")
print("score (test values):",f"{pipe.score(X_test, y_test):.2%}")

# Plot 3D scatter plot:
from mpl_toolkits.mplot3d import Axes3D

# Choose PCA model from pipeline and project data onto the principal components:
X_scaled = pipe.steps[0][1].fit_transform(x) # scaled data
X_trans = pipe.steps[1][1].fit_transform(X_scaled) # Dimension reduction to the main components
y_pred = pipe.predict(x) # Pipeline predicted values...

# Plot 3D scatter diagram:
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_trans[:,0], X_trans[:,1], y, marker="o", c='red', label='actual values')
ax.scatter(X_trans[:, 0], X_trans[:, 1], y_pred, c='blue', label='Predicted')
ax.set_xlabel("PC 1"), ax.set_ylabel("PC 2"), ax.set_zlabel(target)
ax.view_init(azim=-60, elev=20) # position of camera
plt.tight_layout()
plt.legend()
plt.show()

# Plot regression plane witht min/max of the transformed data:
x0 = np.linspace(X_trans[:,0].min(), X_trans[:,0].max(), num=2)
x1 = np.linspace(X_trans[:,1].min(), X_trans[:,1].max(), num=2)
xx0, xx1 = np.meshgrid(x0,x1) # 2x2 - Gitter
X0, X1 = xx0.ravel(), xx1.ravel()
yy = pipe.steps[2][1].predict(np.c_[X0, X1]).ravel() # Prediction values in the regression plane
ax.plot_trisurf(X0, X1, yy, linewidth=0, alpha=0.3)
plt.tight_layout()
plt.show()
