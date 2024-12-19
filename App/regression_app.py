from binary_model import BinaryModel
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, StringVar, Checkbutton, Radiobutton


class RegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Regression Model Builder")
        self.root.geometry("500x400")
        self.model = None
        self.data = None

        tk.Label(root, text="Select Model Type", font=("Arial", 16)).pack(pady=20)
        tk.Button(root, text="Binary Regression", command=lambda: self.initialize_model(BinaryModel)).pack(pady=10)
        tk.Button(root, text="Linear Regression (Coming Soon)", state=tk.DISABLED).pack(pady=10)

    def initialize_model(self, model_class):
        self.model = model_class()
        self.launch_main_menu()

    def launch_main_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Button(self.root, text="Load Data", command=self.load_data).pack(pady=10)
        tk.Button(self.root, text="Select Variables", command=self.select_variables).pack(pady=10)
        tk.Button(self.root, text="Train Model", command=self.train_model).pack(pady=10)
        tk.Button(self.root, text="View Results", command=self.view_results).pack(pady=10)
        tk.Button(self.root, text="Plot ROC Curve", command=self.plot_roc_curve).pack(pady=10)
        tk.Button(self.root, text="Predict", command=self.predict).pack(pady=10)
        tk.Button(self.root, text="Back", command=self.go_back).pack(pady=10)

    def go_back(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Label(self.root, text="Select Model Type", font=("Arial", 16)).pack(pady=20)
        tk.Button(self.root, text="Binary Regression", command=lambda: self.initialize_model(BinaryModel)).pack(
            pady=10)
        tk.Button(self.root, text="Linear Regression (Coming Soon)", state=tk.DISABLED).pack(pady=10)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
        if file_path:
            try:
                self.data = self.model.load_data(file_path)
                messagebox.showinfo("Success", "Data loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def select_variables(self):
        if not hasattr(self, 'data'):
            messagebox.showerror("Error", "No data loaded")
            return

        valid_targets, valid_predictors = self.model.select_variables(self.data)
        if not valid_targets or not valid_predictors:
            messagebox.showerror("Error", "No valid variables found")
            return

        var_window = Toplevel(self.root)
        var_window.title("Select Variables")
        self.target_var = StringVar()

        tk.Label(var_window, text="Select Target Variable").pack()
        for target in valid_targets:
            Radiobutton(var_window, text=target, variable=self.target_var, value=target).pack(anchor="w")

        self.predictor_vars = []
        tk.Label(var_window, text="Select Predictor Variables").pack()
        for predictor in valid_predictors:
            var = StringVar(value=predictor)
            Checkbutton(var_window, text=predictor, variable=var, onvalue=predictor, offvalue="").pack(anchor="w")
            self.predictor_vars.append(var)

        tk.Button(var_window, text="Save", command=lambda: self.save_variables(var_window)).pack(pady=10)

    def save_variables(self, window):
        self.target = self.target_var.get()
        self.predictors = [var.get() for var in self.predictor_vars if var.get()]

        if not self.target:
            messagebox.showerror("Error", "No target variable selected")
            return
        if not self.predictors:
            messagebox.showerror("Error", "No predictor variables selected")
            return

        messagebox.showinfo("Success", f"Selected Target: {self.target}\nSelected Predictors: {', '.join(self.predictors)}")
        window.destroy()

    def train_model(self):
        if not hasattr(self, 'data') or not hasattr(self, 'target') or not hasattr(self, 'predictors'):
            messagebox.showerror("Error", "Ensure data is loaded and variables are selected")
            return
        try:
            self.model.train(self.data, self.target, self.predictors)
            messagebox.showinfo("Success", "Model trained successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def view_results(self):
        if self.model is None or self.model.model is None:
            messagebox.showerror("Error", "No model trained. Please train a model first.")
            return

        results_window = Toplevel(self.root)
        results_window.title("Model Results")
        results_window.geometry("600x700")

        tk.Label(results_window, text="Model Training Results", font=("Arial", 14)).pack(pady=10)

        try:
            summary_text = self.model.get_summary().as_text()

            # Display model summary
            text_widget = tk.Text(results_window, wrap="word", height=15)
            text_widget.insert("1.0", summary_text)
            text_widget.config(state="disabled")
            text_widget.pack(expand=False, fill="x", padx=10, pady=10)

            # Display p-value with significance indication
            p_value = self.model.model.llr_pvalue
            if p_value < 0.05:
                significance_text = f"The model is statistically significant (p = {p_value:.4g})"
                tk.Label(results_window, text=significance_text, font=("Arial", 12), fg="green").pack(pady=10)
            else:
                significance_text = f"Warning: The model is NOT statistically significant (p = {p_value:.4g})"
                tk.Label(results_window, text=significance_text, font=("Arial", 12), fg="red").pack(pady=10)

            # Display regression equation
            coefficients = self.model.model.params
            equation_parts = []
            intercept = coefficients.iloc[0]
            if intercept != 0:
                equation_parts.append(f"{intercept:.4f}")
            for i, predictor in enumerate(self.predictors, start=1):
                coeff = coefficients.iloc[i]
                if coeff > 0:
                    equation_parts.append(f"+ {coeff:.4f} * {predictor}")
                elif coeff < 0:
                    equation_parts.append(f"- {abs(coeff):.4f} * {predictor}")
            equation = "logit(y) = " + " ".join(equation_parts)

            tk.Label(results_window, text=f"Regression Equation:\n{equation}", wraplength=550, justify="left").pack(
                anchor="w", padx=10, pady=5)

            # Display confusion matrix and accuracy
            confusion_matrix = self.model.model.pred_table()
            accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / confusion_matrix.sum()

            tk.Label(results_window, text="Confusion Matrix:", font=("Arial", 12)).pack(anchor="w", padx=10, pady=5)
            tk.Label(results_window, text=f"{confusion_matrix}", justify="left").pack(anchor="w", padx=20)
            tk.Label(results_window, text=f"Accuracy: {accuracy * 100:.2f}%", font=("Arial", 12)).pack(anchor="w",
                                                                                                       padx=10, pady=5)

        except Exception as e:
            tk.Label(results_window, text=f"Error generating model summary: {e}", fg="red").pack(pady=10)

    def plot_roc_curve(self):
        try:
            self.model.plot_results(self.data, self.target, self.predictors)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def predict(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
        if file_path:
            try:
                predictions = self.model.predict(file_path)
                save_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                         filetypes=[("Excel files", "*.xlsx")])
                if save_path:
                    predictions.to_excel(save_path, index=False)
                    messagebox.showinfo("Success", f"Predictions saved to {save_path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
