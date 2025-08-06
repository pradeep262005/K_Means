from flask import Flask, render_template, request
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    label = None
    x_val = y_val = ""

    if request.method == "POST":
        x_val = float(request.form["x_val"])
        y_val = float(request.form["y_val"])
        user_point = np.array([[x_val, y_val]])

        with open("kmeans_model.pkl", "rb") as f:
            kmeans, X = pickle.load(f)

        cluster_number = kmeans.predict(user_point)[0]
        label = f"Cluster {cluster_number}"

        print("User Input:", user_point)
        print("Predicted Cluster:", cluster_number)

        # Plot clusters
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=30, cmap="viridis", alpha=0.6)
        centers = kmeans.cluster_centers_

        # Draw circles around cluster centers
        for center in centers:
            circle = plt.Circle(center, 2.0, color='red', fill=False, linewidth=2)
            plt.gca().add_patch(circle)

        # Mark user input
        plt.scatter(user_point[0][0], user_point[0][1], c='black', s=150,
                    edgecolors='white', marker='X', label='Your Point')
        plt.legend()

        plt.title("K-Means Clustering with Your Input")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        # Save plot
        if not os.path.exists("static"):
            os.makedirs("static")
        plt.savefig("static/cluster.png")
        plt.close()

    return render_template("index.html", label=label, x_val=x_val, y_val=y_val)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
