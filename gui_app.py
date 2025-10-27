import tkinter as tk
from tkinter import messagebox
import joblib

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_news():
    text = input_box.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Empty Input", "Please enter news text.")
        return
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    color = "green" if pred == "REAL" else "red"
    result_label.config(text=f"Prediction: {pred}", fg=color)

root = tk.Tk()
root.title("Fake News Detector")
root.geometry("600x400")

tk.Label(root, text="Enter News Text:", font=("Arial", 14)).pack(pady=10)
input_box = tk.Text(root, height=10, width=70)
input_box.pack()

tk.Button(root, text="Check News", command=predict_news, font=("Arial", 12)).pack(pady=10)
result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack(pady=20)

root.mainloop()
