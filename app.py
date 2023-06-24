import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def random_forest_page():
    st.title("Random Forest Classifier")
    st.write("Adjust the parameters of a Random Forest Classifier and see the impact.")

    data = load_iris()
    X, y = data.data[:, :2], data.target  # Only using the first two features for visualization
    clf = RandomForestClassifier()

    n_estimators = st.slider("Number of Estimators", 1, 100, value=10)
    max_depth = st.slider("Maximum Depth", 1, 10, value=5)

    clf.set_params(n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(X, y)
    accuracy = clf.score(X, y)
    st.write("Accuracy:", accuracy)

    # Decision boundaries
    h = 0.02  # Step size of the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    cmap = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["#FF0000", "#00FF00", "#0000FF"]), edgecolor="k")
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])
    plt.title("Decision Boundaries")
    st.subheader("Decision Boundaries")
    st.pyplot(plt)


def svm_page():
    st.title("Support Vector Classifier")
    st.write("Adjust the parameters of a Support Vector Classifier and see the impact.")

    data = load_iris()
    X, y = data.data[:, :2], data.target  # Only using the first two features for visualization
    clf = SVC()

    C = st.slider("C (Regularization Parameter)", 0.01, 10.0, step=0.01, value=1.0)
    gamma = st.slider("Gamma (Kernel Coefficient)", 0.01, 10.0, step=0.01, value=1.0)

    clf.set_params(C=C, gamma=gamma)
    clf.fit(X, y)
    accuracy = clf.score(X, y)
    st.write("Accuracy:", accuracy)

    # Decision boundaries
    h = 0.02  # Step size of the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    cmap = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["#FF0000", "#00FF00", "#0000FF"]), edgecolor="k")
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])
    plt.title("Decision Boundaries")
    st.subheader("Decision Boundaries")
    st.pyplot(plt)


def knn_page():
    st.title("K-Nearest Neighbors Classifier")
    st.write("Adjust the parameters of a K-Nearest Neighbors Classifier and see the impact.")

    data = load_iris()
    X, y = data.data[:, :2], data.target  # Only using the first two features for visualization
    clf = KNeighborsClassifier()

    n_neighbors = st.slider("Number of Neighbors", 1, 10, value=5)

    clf.set_params(n_neighbors=n_neighbors)
    clf.fit(X, y)
    accuracy = clf.score(X, y)
    st.write("Accuracy:", accuracy)

    # Decision boundaries
    h = 0.02  # Step size of the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    cmap = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["#FF0000", "#00FF00", "#0000FF"]), edgecolor="k")
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])
    plt.title("Decision Boundaries")
    st.subheader("Decision Boundaries")
    st.pyplot(plt)
def homepage():
    st.title("Machine Learning Classifier Explorer")
    st.write("Welcome to the Machine Learning Classifier Explorer app!")
    st.write("This interactive platform allows you to explore different classification algorithms using the Iris dataset.")
    st.write("Adjust the parameters of models like Random Forest, Support Vector Machine, and K-Nearest Neighbors to see how decision boundaries change and their impact on model accuracy.")
    st.write("It's an educational tool that helps you understand the behavior of different machine learning models.")
    st.write("Let's dive in and explore the world of classification algorithms together!")


def main():
    st.sidebar.title("Model Selection")
    model_choice = st.sidebar.radio("", ("Homepage", "Random Forest", "Support Vector Machine", "K-Nearest Neighbors"))

    if model_choice == "Homepage":
        homepage()
    elif model_choice == "Random Forest":
        st.subheader("Random Forest Classifier")
        random_forest_page()
    elif model_choice == "Support Vector Machine":
        st.subheader("Support Vector Classifier")
        svm_page()
    elif model_choice == "K-Nearest Neighbors":
        st.subheader("K-Nearest Neighbors Classifier")
        knn_page()

st.info("Created by Yash Thapliyal, 2023\n[GitHub Profile](https://github.com/AlphaIdylSaythTG)")        





if __name__ == "__main__":
    main()
