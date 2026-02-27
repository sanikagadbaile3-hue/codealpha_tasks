import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import os

class ModernCard(tk.Canvas):
    def __init__(self, parent, bg="#ffffff", radius=25, **kwargs):
        super().__init__(parent, bg=parent["bg"], highlightthickness=0, **kwargs)
        self.radius = radius
        self.card_bg = bg
        self.bind("<Configure>", self.draw)

    def draw(self, event=None):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        r = self.radius
        self.create_rounded_rect(2, 2, w-2, h-2, r, fill=self.card_bg)

    def create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y1+r, x2, y2-r, x2, y2-r, x2, y2, x2-r, y2, x2-r, y2, x1+r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y2-r, x1, y1+r, x1, y1+r, x1, y1]
        return self.create_polygon(points, **kwargs, smooth=True)

class DiseasePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical AI Dashboard")
        self.root.geometry("1100x850")
        self.root.configure(bg="#f4f7f9")
        
        # Premium Color Palette
        self.COLORS = {
            "bg": "#f4f7f9",
            "sidebar": "#1b1f23",
            "nav_active": "#343a40",
            "card": "#ffffff",
            "primary": "#007bff",
            "text_main": "#2d3436",
            "text_muted": "#636e72",
            "heart": "#ff4757",
            "diabetes": "#1e90ff",
            "cancer": "#ed4c67"
        }
        
        self.current_disease = "Heart Disease"
        self.models = {}
        self.scalers = {}
        self.inputs = {}
        
        self.load_models()
        self.setup_layout()
        self.switch_disease("Heart Disease")

    def load_models(self):
        for d in ['heart', 'diabetes', 'breast_cancer']:
            try:
                m_path = f'models/{d}_model.joblib'
                s_path = f'models/{d}_scaler.joblib'
                if os.path.exists(m_path):
                    self.models[d] = joblib.load(m_path)
                    self.scalers[d] = joblib.load(s_path)
            except: pass

    def setup_layout(self):
        # Sidebar Navigation
        self.sidebar = tk.Frame(self.root, bg=self.COLORS["sidebar"], width=280)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        # Dashboard Brand
        brand_frame = tk.Frame(self.sidebar, bg=self.COLORS["sidebar"], pady=50)
        brand_frame.pack(fill=tk.X)
        tk.Label(brand_frame, text="MED.AI", font=("Arial Black", 24), bg=self.COLORS["sidebar"], fg="white").pack()
        tk.Label(brand_frame, text="DASHBOARD v2.0", font=("Segoe UI", 8, "bold"), bg=self.COLORS["sidebar"], fg="#4b5563").pack()

        # Navigation Menu
        self.nav_items = {}
        menu_items = [
            ("Heart Disease", "❤️"),
            ("Diabetes", "💧"),
            ("Breast Cancer", "🎗️")
        ]
        
        for name, icon in menu_items:
            f = tk.Frame(self.sidebar, bg=self.COLORS["sidebar"])
            f.pack(fill=tk.X, pady=5)
            
            btn = tk.Button(f, text=f"  {icon}  {name}", font=("Segoe UI", 11, "bold"),
                          bg=self.COLORS["sidebar"], fg="#cbd5e0", relief="flat", anchor="w",
                          padx=30, pady=15, cursor="hand2", activebackground=self.COLORS["nav_active"], 
                          activeforeground="white", command=lambda n=name: self.switch_disease(n))
            btn.pack(fill=tk.X)
            self.nav_items[name] = btn

        # Main Workspace
        self.workspace = tk.Frame(self.root, bg=self.COLORS["bg"], padx=50, pady=50)
        self.workspace.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Header Info
        self.header = tk.Frame(self.workspace, bg=self.COLORS["bg"])
        self.header.pack(fill=tk.X)
        
        self.title = tk.Label(self.header, text="Clinical Analysis", font=("Segoe UI", 28, "bold"), bg=self.COLORS["bg"], fg=self.COLORS["text_main"])
        self.title.pack(anchor="w")
        self.subtitle = tk.Label(self.header, text="Ready for diagnostic input...", font=("Segoe UI", 12), bg=self.COLORS["bg"], fg=self.COLORS["text_muted"])
        self.subtitle.pack(anchor="w", pady=(5, 30))

        # The Card Container
        self.main_card = ModernCard(self.workspace, bg="white", height=500)
        self.main_card.pack(fill=tk.BOTH, expand=True)
        
        self.content = tk.Frame(self.main_card, bg="white", padx=40, pady=40)
        self.content.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.grid_container = tk.Frame(self.content, bg="white")
        self.grid_container.pack(fill=tk.BOTH, expand=True)

        # Action Area
        self.action_bar = tk.Frame(self.workspace, bg=self.COLORS["bg"], pady=30)
        self.action_bar.pack(fill=tk.X)

        self.run_btn = tk.Button(self.action_bar, text="RUN ANALYSIS", font=("Segoe UI", 12, "bold"), 
                               bg=self.COLORS["primary"], fg="white", relief="flat", 
                               padx=40, pady=15, cursor="hand2", command=self.predict)
        self.run_btn.pack(side=tk.LEFT)

        self.reset_btn = tk.Button(self.action_bar, text="FLASH CLEAR", font=("Segoe UI", 11), 
                                 bg="white", fg=self.COLORS["text_muted"], relief="flat", 
                                 padx=20, pady=15, cursor="hand2", command=self.clear)
        self.reset_btn.pack(side=tk.LEFT, padx=20)

        # Status Badge
        self.status_card = tk.Frame(self.action_bar, bg="#dae1e7", padx=20, pady=10)
        self.status_card.pack(side=tk.RIGHT)
        self.status_lbl = tk.Label(self.status_card, text="STATUS: IDLE", font=("Segoe UI", 9, "bold"), bg="#dae1e7", fg="#4b5563")
        self.status_lbl.pack()

    def switch_disease(self, name):
        self.current_disease = name
        theme_colors = {"Heart Disease": self.COLORS["heart"], "Diabetes": self.COLORS["diabetes"], "Breast Cancer": self.COLORS["cancer"]}
        active_color = theme_colors.get(name, self.COLORS["primary"])
        
        # UI Updates
        self.title.config(text=name)
        self.run_btn.config(bg=active_color)
        
        # Update Nav Styles
        for n, btn in self.nav_items.items():
            if n == name:
                btn.config(bg=self.COLORS["nav_active"], fg="white")
            else:
                btn.config(bg=self.COLORS["sidebar"], fg="#cbd5e0")

        # Field Reconstruction
        for w in self.grid_container.winfo_children(): w.destroy()
        self.inputs = {}
        
        if name == "Heart Disease":
            fields = [("Age", "age"), ("Sex (1:M,0:F)", "sex"), ("CP Type", "cp"), ("BP", "trestbps"), ("Chol", "chol"), 
                      ("Sugar > 120", "fbs"), ("ECG", "restecg"), ("Max HR", "thalach"), ("Angina", "exang"), 
                      ("ST Depr", "oldpeak"), ("Slope", "slope"), ("CA", "ca"), ("Thal", "thal")]
        elif name == "Diabetes":
            fields = [("Pregnancies", "Pregnancies"), ("Glucose", "Glucose"), ("Blood Pr", "BloodPressure"), 
                      ("Skin", "SkinThickness"), ("Insulin", "Insulin"), ("BMI", "BMI"), ("Pedigree", "DiabetesPedigreeFunction"), ("Age", "Age")]
        else:
            fields = [("Radius", "radius_mean"), ("Texture", "texture_mean"), ("Perimeter", "perimeter_mean"), ("Area", "area_mean"), 
                      ("Smoothness", "smoothness_mean"), ("Compact", "compactness_mean"), ("Concavity", "concavity_mean"), 
                      ("Points", "concave_points_mean"), ("Symmetry", "symmetry_mean")]

        for i, (label, key) in enumerate(fields):
            r, c = i // 4, i % 4
            cell = tk.Frame(self.grid_container, bg="white", padx=10, pady=15)
            cell.grid(row=r, column=c, sticky="nsew")
            
            tk.Label(cell, text=label.upper(), font=("Segoe UI", 8, "bold"), bg="white", fg=self.COLORS["text_muted"]).pack(anchor="w")
            e = tk.Entry(cell, font=("Segoe UI", 12), bg="#f8f9fa", relief="flat", highlightbackground="#e2e8f0", highlightthickness=1)
            e.pack(fill=tk.X, pady=5, ipady=5)
            self.inputs[key] = e

        for i in range(4): self.grid_container.columnconfigure(i, weight=1)

    def clear(self):
        for e in self.inputs.values(): e.delete(0, tk.END)
        self.status_lbl.config(text="STATUS: IDLE", fg="#4b5563")

    def predict(self):
        d_map = {"Heart Disease": "heart", "Diabetes": "diabetes", "Breast Cancer": "breast_cancer"}
        d_key = d_map[self.current_disease]
        
        if d_key not in self.models:
            messagebox.showerror("Error", "Model files missing!")
            return
            
        try:
            vals = [float(self.inputs[k].get()) for k in self.inputs]
            if d_key == 'breast_cancer':
                x = np.zeros(30); x[:len(vals)] = vals
                data = x.reshape(1, -1)
            else:
                data = np.array(vals).reshape(1, -1)
            
            sc_data = self.scalers[d_key].transform(data)
            pred = self.models[d_key].predict(sc_data)[0]
            prob = self.models[d_key].predict_proba(sc_data)[0][1]
            
            txt = "DISEASE DETECTED" if pred == 1 else "NO INDICATION"
            clr = self.COLORS["danger"] if pred == 1 else self.COLORS["success"]
            
            self.status_lbl.config(text=f"{txt} ({prob*100:.1f}%)", fg=clr)
            
            if pred == 1:
                messagebox.showwarning("Insight", f"Clinical markers suggest {self.current_disease} risk. Immediate specialist consultation recommended.")
            else:
                messagebox.showinfo("Insight", "Markers within benign range. Continue regular screening.")
                
        except Exception as e:
            messagebox.showerror("Validation", f"Invalid biometric data: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DiseasePredictionApp(root)
    root.mainloop()
