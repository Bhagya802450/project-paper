import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import re

# Load trained model and scaler
model = joblib.load("advanced_ddos_ensemble_model.pkl")
scaler = joblib.load("advanced_scaler.pkl")

# 🔐 AES Encryption
def encrypt_text(text: str, key: bytes) -> bytes:
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(text.encode()) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(padded_data) + encryptor.finalize()

# ❌ Fix: Remove Unicode characters not supported by Tcl
def remove_high_unicode(text):
    return re.sub(r'[^\u0000-\uFFFF]', '', str(text))

# GUI Class
class DDoSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Healthcare Cybersecurity - DDoS Detection")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f4f7")

        tk.Label(root, text="Healthcare Cybersecurity DDoS Detection", font=("Arial", 22, "bold"), bg="#f0f4f7", fg="#003366").pack(pady=20)

        self.upload_btn = tk.Button(root, text="Upload CSV", command=self.upload_file, font=("Arial", 14), bg="#007acc", fg="white")
        self.upload_btn.pack()

        self.tree_frame = tk.Frame(root)
        self.tree_frame.pack(pady=20, fill='both', expand=True)

        self.tree_scroll = tk.Scrollbar(self.tree_frame)
        self.tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree = ttk.Treeview(self.tree_frame, yscrollcommand=self.tree_scroll.set)
        self.tree.pack(fill='both', expand=True)
        self.tree_scroll.config(command=self.tree.yview)

        self.graph_btn = tk.Button(root, text="Show Attack Graph", command=self.show_attack_graph, font=("Arial", 12), bg="#28a745", fg="white")
        self.graph_btn.pack(pady=5)

        self.download_btn = tk.Button(root, text="Download Results", state='disabled', command=self.download_results, font=("Arial", 12), bg="#17a2b8", fg="white")
        self.download_btn.pack(pady=5)

        self.df = None

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)
            df = df.applymap(remove_high_unicode)  # ✅ Remove problematic characters
            if df.empty:
                raise ValueError("CSV is empty.")
            messagebox.showinfo("Upload", "File uploaded successfully!")
            self.process_dataframe(df)
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file:\n{str(e)}")

    def process_dataframe(self, df):
        self.df = df.copy()

        # 🔒 Encrypt patient data (optional)
        if 'patient_info' in self.df.columns:
            aes_key = os.urandom(16)
            self.df['encrypted_info'] = self.df['patient_info'].apply(lambda x: encrypt_text(str(x), aes_key))

        df_clean = self.df.copy()

        # 🔠 Encode categorical columns except label
        for col in df_clean.select_dtypes(include='object').columns:
            if col != 'label':
                df_clean[col] = LabelEncoder().fit_transform(df_clean[col])

        if 'label' in df_clean.columns:
            X = df_clean.drop(['label'], axis=1)
        else:
            X = df_clean

        # ⚖️ Scale
        X_scaled = scaler.transform(X)

        # 🤖 Predict
        predictions = model.predict(X_scaled)

        self.df['Prediction'] = predictions
        self.df['Status'] = self.df['Prediction'].apply(lambda x: 'Attack' if x == 1 else 'Normal')

        self.display_dataframe(self.df[['Prediction', 'Status']].value_counts().reset_index(name='Count'))
        self.download_btn.config(state='normal')

    def display_dataframe(self, df):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree["columns"] = list(df.columns)
        self.tree["show"] = "headings"

        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)

        for _, row in df.iterrows():
            self.tree.insert("", "end", values=list(row))

    def show_attack_graph(self):
        import matplotlib.pyplot as plt

        if self.df is not None and 'Status' in self.df.columns:
            count = self.df['Status'].value_counts()
            count.plot(kind='bar', color=['green', 'red'])
            plt.title("Attack vs Normal Packets")
            plt.xlabel("Status")
            plt.ylabel("Count")
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showwarning("No Data", "Please upload and process a CSV file first.")

    def download_results(self):
        if self.df is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV File", "*.csv")])
            if file_path:
                self.df.to_csv(file_path, index=False)
                messagebox.showinfo("Download Complete", f"Results saved to:\n{file_path}")

# 🔁 Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = DDoSApp(root)
    root.mainloop()
