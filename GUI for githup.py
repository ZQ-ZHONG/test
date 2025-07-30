import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

# -------------------------- Global configuration (Complete ALE data) --------------------------
Sf = np.pi * (0.125 * 1e-3) ** 2  # Fiber cross-sectional area (m²)
default_values = {
    'alpha1': 7.5, 'alpha2': 25.0, 'alpha_f': 1.025, 'E1': 175.0, 'E2': 175.0,
    'b': 17.5, 'h1': 1.75, 'h2': 1.75, 'Ls': 150.0, 'Ef': 42.5, 'theta': 45.0
}

# Parameter ranges and unit information
param_info = {
    'alpha1': {'unit': '10⁻⁶/°C'},
    'alpha2': {'unit': '10⁻⁶/°C'},
    'alpha_f': {'unit': '10⁻⁶/°C'},
    'E1': {'unit': 'GPa'},
    'E2': {'unit': 'GPa'},
    'b': {'unit': 'mm'},
    'h1': {'unit': 'mm'},
    'h2': {'unit': 'mm'},
    'Ls': {'unit': 'mm'},
    'Ef': {'unit': 'GPa'},
    'theta': {'unit': '°'}
}

# -------------------------- Complete ALE data (21 data points provided by the user) --------------------------
ale_data = {
    'sigma': (
        np.array([0.000925854, 0.01872258, 0.037585749, 0.053959005, 0.071581595, 0.08930732, 0.107349768, 0.1265862,
                  0.142167053, 0.159347363, 0.17652097, 0.193115093, 0.210045842, 0.227438399, 0.243816612, 0.262110232,
                  0.280860887, 0.297265403, 0.314858492, 0.331502379, 0.349887492]),
        np.array(
            [5.419078111, 3.001156567, 1.596096038, 0.923685371, 0.472414373, 0.166990518, -0.060541541, -0.254176335,
             -0.386932076, -0.510958926, -0.614617833, -0.700656624, -0.779920587, -0.853731574, -0.914149464,
             -0.972431635, -1.02550611, -1.067789209, -1.109583595, -1.146003739, -1.18242173])
    ),
    'omega': (
        np.array(
            [1.674144894, 4.516624784, 7.466730864, 10.36909652, 13.34296797, 16.32824095, 19.22465282, 22.07670573,
             24.60602831, 27.52405872, 30.4350863, 33.44388975, 36.26549852, 39.01875958, 42.09591423, 44.99302402,
             47.86912563, 50.6869534, 53.74202221, 57.0170254, 59.97743448]),
        np.array([-1.647964155, -1.410231432, -1.172779402, -0.950279987, -0.735621249, -0.535409456, -0.35704436,
                  -0.197315326, -0.068734607, 0.065217623, 0.185107613, 0.29700228, 0.393222855, 0.480895588,
                  0.573244372, 0.655777397, 0.734117614, 0.807819808, 0.884696843, 0.964148784, 1.034129197])
    ),
    'delta': (
        np.array([0.166672234, 0.460459904, 0.743969568, 1.02513928, 1.312655882, 1.608768309, 1.890803292, 2.181421176,
                  2.453561636, 2.794170323, 3.069606193, 3.356592271, 3.642447844, 3.944301279, 4.241905347,
                  4.561718942, 4.840855219, 5.130040955, 5.397335834, 5.681503592, 5.99829915]),
        np.array(
            [-1.057118755, -0.586259883, -0.192569029, 0.110397149, 0.317454237, 0.433417644, 0.472764011, 0.467555614,
             0.440978203, 0.38977145, 0.334164161, 0.264132913, 0.185622793, 0.097639684, 0.010114092, -0.083612006,
             -0.165373255, -0.247596521, -0.320274446, -0.395094629, -0.476113428])
    ),
    'kappa': (
        np.array([3.553744958, 12.45830776, 19.68507372, 27.54220623, 34.68880877, 41.4357028, 48.36070598, 56.04374124,
                  62.54578505, 69.80867356, 76.81124005, 83.89707749, 91.26263352, 97.86058374, 105.5156706,
                  114.8139173, 126.3825016, 141.3500158, 168.6140112, 210.714883, 378.6861807]),
        np.array([-1.224837706, -0.838573501, -0.557379678, -0.292112992, -0.092110605, 0.058654606, 0.175368793,
                  0.263126894, 0.307798922, 0.332100359, 0.336306494, 0.327195635, 0.30870806, 0.287739269, 0.260945504,
                  0.227375561, 0.186335695, 0.136534573, 0.059559958, -0.027952923, -0.234782918])
    ),
    'theta': (
        np.array(
            [15.02505549, 18.16130717, 21.15550176, 24.26025427, 27.16280651, 29.94718508, 33.33399685, 36.29425843,
             39.28974645, 42.16299751, 44.90407272, 47.72414355, 50.71672954, 53.72654655, 56.63749812, 59.75102833,
             62.72532547, 65.93786354, 68.90175372, 72.01500013, 74.9763124]),
        np.array([-0.556470391, -0.489077609, -0.428690743, -0.370038405, -0.318607856, -0.272070926, -0.218670808,
                  -0.174278114, -0.130772163, -0.089584578, -0.049938262, -0.007681948, 0.040426101, 0.094312586,
                  0.153780296, 0.227311939, 0.308209911, 0.407405793, 0.508988589, 0.624549798, 0.740896791])
    ),
    'eta': (
        np.array(
            [1.931625104, 1.90036552, 1.865208745, 1.823377045, 1.779425821, 1.732971866, 1.670962896, 1.611975207,
             1.547907092, 1.482476498, 1.416579322, 1.345401569, 1.266309791, 1.183279417, 1.099868486,
             1.007516946, 0.916513459, 0.815454262, 0.719936504, 0.617535991, 0.518436725]),
        np.array([-0.556470391, -0.489077609, -0.428690743, -0.370038405, -0.318607856, -0.272070926, -0.218670808,
                  -0.174278114, -0.130772163, -0.089584578, -0.049938262, -0.007681948, 0.040426101, 0.094312586,
                  0.153780296, 0.227311939, 0.308209911, 0.407405793, 0.508988589, 0.624549798, 0.740896791])
    ),
    'gamma': (
        np.array([0.025181061, 0.050161272, 0.072361791, 0.096914364, 0.121918063, 0.14493611, 0.167332836, 0.190641316,
                  0.215430322, 0.239425879, 0.262724801, 0.288250153, 0.311953981, 0.335424121, 0.360090592, 0.38096923,
                  0.405730012, 0.429113152, 0.45291264, 0.477331821, 0.499911996]),
        np.array([0.773952705, 0.692736995, 0.619976689, 0.539126751, 0.456625578, 0.380698879, 0.306942684, 0.23037449,
                  0.149215152, 0.070987862, -0.004581469, -0.086859408, -0.162745993, -0.237397823, -0.315389927,
                  -0.381103556, -0.458774853, -0.531951758, -0.606323998, -0.682550672, -0.75295833])
    ),
    'beta': (
        np.array(
            [0.400287956, 0.495956088, 0.597901319, 0.705556146, 0.813468742, 0.914887606, 1.022784835, 1.128602819,
             1.223586856, 1.328203977, 1.43366649, 1.528851658, 1.647460197, 1.749694129, 1.856291466, 1.968925621,
             2.071707143, 2.176873069, 2.284656912, 2.386033328, 2.499088767]),
        np.array(
            [-0.001864255, 0.028576824, 0.05305582, 0.070825054, 0.081249119, 0.085171217, 0.083998668, 0.078354129,
             0.07015484, 0.058331394, 0.044025216, 0.029458327, 0.009533958, -0.0089437, -0.029289772, -0.051820579,
             -0.073196259, -0.095798791, -0.119686133, -0.142804288, -0.169330789])
    ),
    'gamma_f': (
        np.array(
            [0.013821655, 0.021291587, 0.027671561, 0.034083711, 0.040515592, 0.046881656, 0.053731168, 0.060318574,
             0.067298672, 0.07476442, 0.081658298, 0.088584207, 0.095366264, 0.102402445, 0.108721436, 0.115867559,
             0.122721245, 0.129261953, 0.135982105, 0.143029244, 0.149997658]),
        np.array(
            [0.077171869, 0.071344511, 0.065915664, 0.060064858, 0.053815878, 0.047268764, 0.039836031, 0.032324744,
             0.024001225, 0.014715252, 0.005816093, -0.003419208, -0.012740563, -0.022702565, -0.031909956,
             -0.042637482, -0.053267589, -0.063759912, -0.074938, -0.087145346, -0.099754269])
    )
}

var_to_greek = {
    'sigma': 'σ', 'omega': 'ω', 'delta': 'δ', 'kappa': 'κ', 'theta': 'θ',
    'eta': 'η', 'gamma': 'γ', 'beta': 'β', 'gamma_f': 'γ_f'
}
user_order = ['sigma', 'omega', 'delta', 'kappa', 'theta', 'eta', 'gamma', 'beta', 'gamma_f']  # Subplot order


# -------------------------- Calculation functions --------------------------
def calculate_intermediate(params):
    """Calculate intermediate variables"""
    alpha1 = params['alpha1'] * 1e-6
    alpha2 = params['alpha2'] * 1e-6
    alpha_f = params['alpha_f'] * 1e-6
    E1 = params['E1'] * 1e9
    E2 = params['E2'] * 1e9
    b = params['b'] * 1e-3
    h1 = params['h1'] * 1e-3
    h2 = params['h2'] * 1e-3
    Ls = params['Ls'] * 1e-3
    Ef = params['Ef'] * 1e9
    theta_rad = np.radians(params['theta'])

    gamma = alpha1 / alpha2
    gamma_f = alpha_f / alpha2
    beta = E1 / E2
    omega = b / h2
    delta = h1 / h2
    La = Ls / (2 * np.cos(theta_rad))
    kappa = La / h2
    sigma = (Ef * Sf) / (E2 * h2 ** 2)
    eta = Ls / La

    return {
        'gamma': gamma, 'gamma_f': gamma_f, 'beta': beta, 'omega': omega, 'delta': delta,
        'kappa': kappa, 'sigma': sigma, 'eta': eta, 'theta': params['theta']
    }


def calculate_MM(intermediate):
    """Calculate the MM value"""
    gamma = intermediate['gamma']
    gamma_f = intermediate['gamma_f']
    beta = intermediate['beta']
    omega = intermediate['omega']
    delta = intermediate['delta']
    kappa = intermediate['kappa']
    sigma = intermediate['sigma']
    eta = intermediate['eta']
    theta_rad = np.radians(intermediate['theta'])

    term1 = 1 + beta * delta ** 3
    term2 = -1 - gamma_f * eta - beta * delta * (
            delta * (3 + 4 * delta) +
            gamma * (4 + 3 * delta + beta * delta ** 3) +
            gamma * (4 + delta * (6 + delta * (4 + beta * delta))) * eta
    )
    term3 = 2 * (1 + beta * delta * (
            6 + delta * (9 + 4 * delta) +
            gamma * (-2 - 3 * delta + beta * delta ** 3)
    )) * np.cos(theta_rad)
    term4 = (1 + beta * delta * (
            delta * (3 + 4 * delta) +
            gamma * (4 + 3 * delta + beta * delta ** 3)
    )) * np.cos(2 * theta_rad)
    term5 = -6 * beta * (-1 + gamma) * delta * (1 + delta) * kappa * np.sin(theta_rad)
    mm_num = term1 * omega * (term2 + term3 + term4 + term5)

    term6 = 8 * (1 + beta * delta) * (1 + beta * delta ** 3) * kappa ** 2 * sigma * np.sin(theta_rad) ** 2
    term7 = 1 + beta * delta * (4 + delta * (6 + delta * (4 + beta * delta)))
    term8 = (1 + beta * delta ** 3) * eta * omega + 3 * (1 + delta) * kappa * sigma * np.sin(2 * theta_rad)
    mm_den = term6 + term7 * term8

    return mm_num, mm_den


def calculate_K_dimless(mm_num, mm_den):
    """Calculate K_dimless"""
    if mm_den == 0:
        raise ZeroDivisionError("MM denominator cannot be zero!")
    return mm_num / mm_den


# -------------------------- GUI class (complete revision) --------------------------
class ALEGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Proposed by Tianran Han and Ziqin Zhong")
        # Increase window height to accommodate larger title
        self.root.geometry("1300x880")

        # Adjust layout weights - Left input panel: 1/3, Right plot panel: 2/3
        self.root.grid_columnconfigure(0, weight=1)  # Left column (1/3 width)
        self.root.grid_columnconfigure(1, weight=2)  # Right column (2/3 width)
        self.root.grid_rowconfigure(0, weight=1)  # Row weight, fill entire window

        # -------------------------- Top title bar (newly added) --------------------------
        self.title_frame = ttk.Frame(root)
        self.title_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

        # Add large font title
        self.title_label = ttk.Label(
            self.title_frame,
            text="Proposed by Tianran Han and Ziqin Zhong",
            font=('Times New Roman', 24, 'bold'),
            foreground='#003366'
        )
        self.title_label.pack(pady=10)

        # -------------------------- Left: Input parameters panel --------------------------
        self.input_frame = ttk.Frame(root, padding=15)
        self.input_frame.grid(row=1, column=0, sticky='ns')  # Move to second row

        # Add title (left input panel)
        title_label = ttk.Label(
            self.input_frame,
            text="Input variables",
            font=('Times New Roman', 24, 'bold'),
            foreground='#003366'
        )
        title_label.pack(pady=(0, 15))

        self.inputs = {}
        input_params = [
            ('α₁:', 'alpha1'), ('α₂:', 'alpha2'), ('α_f:', 'alpha_f'),
            ('E₁:', 'E1'), ('E₂:', 'E2'), ('b:', 'b'), ('h₁:', 'h1'),
            ('h₂:', 'h2'), ('Lₛ:', 'Ls'), ('E_f:', 'Ef'), ('θ:', 'theta')
        ]

        # Create input boxes and labels
        for label_text, param_name in input_params:
            frame = ttk.Frame(self.input_frame)
            frame.pack(fill=tk.X, pady=18)  # Adjust row spacing to 10 pixels

            # Parameter label
            ttk.Label(
                frame,
                text=label_text,
                font=('Times New Roman', 18),
                width=4,
                anchor='e'
            ).pack(side=tk.LEFT, padx=(0, 5))

            # Input box
            entry = ttk.Entry(
                frame,
                width=10,
                font=('Times New Roman', 16)
            )
            entry.insert(0, str(default_values[param_name]))
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.inputs[param_name] = entry

            # Unit label
            unit_info = param_info[param_name]
            unit_label = ttk.Label(
                frame,
                text=f"{unit_info['unit']}",
                font=('Times New Roman', 16),
                width=10,
                anchor='w',
                foreground='#666666'
            )
            unit_label.pack(side=tk.LEFT, padx=(5, 0))

        # Create button frame
        self.button_frame = ttk.Frame(self.input_frame)
        self.button_frame.pack(fill=tk.X, pady=1, expand=True)

        # Calculate button
        self.calc_btn = ttk.Button(
            self.button_frame,
            text="Calculate",
            command=self.on_calculate,
            style='TButton'
        )
        self.calc_btn.pack(side=tk.LEFT, padx=2, ipadx=0, fill=tk.X, expand=True)

        # Default button
        self.default_btn = ttk.Button(
            self.button_frame,
            text="Default",
            command=self.on_default,
            style='TButton'
        )
        self.default_btn.pack(side=tk.LEFT, padx=2, ipadx=0, fill=tk.X, expand=True)

        # Cancel button
        self.cancel_btn = ttk.Button(
            self.button_frame,
            text="Cancel",
            command=self.on_cancel,
            style='TButton'
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=2, ipadx=0, fill=tk.X, expand=True)

        # -------------------------- Right: Subplot panel --------------------------
        self.plot_frame = ttk.Frame(root, padding=10)
        self.plot_frame.grid(row=1, column=1, sticky='nsew')  # Move to second row

        # Add title (right plot panel)
        plot_title = ttk.Label(
            self.plot_frame,
            text="ALE plots with calculated values",
            font=('Times New Roman', 24, 'bold'),
            foreground='#006633'
        )
        plot_title.pack(pady=(0, 10))

        # Create 3x3 subplots
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.subplots_adjust(hspace=0.5, wspace=0.4)  # Adjust subplot spacing

        # Manually set the position of each subplot [left, bottom, width, height]
        positions = [
            [0.08, 0.73, 0.21, 0.20],  # First row
            [0.415, 0.73, 0.21, 0.20],
            [0.75, 0.73, 0.21, 0.20],
            [0.08, 0.43, 0.21, 0.20],
            [0.415, 0.43, 0.21, 0.20],
            [0.75, 0.43, 0.21, 0.20],
            [0.08, 0.13, 0.21, 0.20],
            [0.415, 0.13, 0.21, 0.20],
            [0.75, 0.13, 0.21, 0.20]
        ]

        self.axes = {}
        for i, var in enumerate(user_order):
            ax = self.fig.add_axes(positions[i])
            self.axes[var] = ax

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize variable label dictionary
        self.var_labels = {}
        self.k_dimless_text = None  # Text object for displaying K_dimless value

        # Configure each subplot
        for var in user_order:
            ax = self.axes[var]

            # Plot ALE curve (pink solid line)
            x_data, y_data = ale_data[var]
            ax.plot(x_data, y_data, linewidth=2, color="#D96D71")

            # Set ticks
            if var == 'sigma':
                ax.set_xticks(np.arange(0.0, 0.401, 0.1))  # Include 0.4
                ax.set_yticks(np.arange(-2, 9.1, 2))  # Include 8
            elif var == 'omega':
                ax.set_xticks(np.arange(0, 81, 20))  # Include 80
                ax.set_yticks(np.arange(-2.0, 1.6 + 1e-6, 0.5))  # Include 1.5
            elif var == 'delta':
                ax.set_xticks(np.arange(0, 9.1, 2))  # Include 8
                ax.set_yticks(np.arange(-1.5, 1.1 + 1e-6, 0.5))  # Include 1.0
            elif var == 'kappa':
                ax.set_xticks(np.arange(0, 401, 100))  # Include 400
                ax.set_yticks(np.arange(-1.5, 0.6 + 1e-6, 0.5))  # Include 0.5
            elif var == 'theta':
                ax.set_xticks(np.arange(0, 81, 20))  # Include 80
                ax.set_yticks(np.arange(-1.0, 1.1 + 1e-6, 0.5))  # Include 1.0
            elif var == 'eta':
                ax.set_xticks(np.arange(0.0, 2.1 + 1e-6, 0.5))  # Include 2.0
                ax.set_yticks(np.arange(-1.0, 1.1 + 1e-6, 0.5))  # Include 1.0
            elif var == 'gamma':
                ax.set_xticks(np.arange(0.0, 0.61 + 1e-6, 0.2))  # Include 0.6
                ax.set_yticks(np.arange(-1.0, 1.1 + 1e-6, 0.5))  # Include 1.0
            elif var == 'beta':
                ax.set_xticks(np.arange(0.0, 3.1 + 1e-6, 0.5))  # Include 3.0
                ax.set_yticks(np.arange(-0.20, 0.16 + 1e-6, 0.05))  # Include 0.15
            elif var == 'gamma_f':
                ax.set_xticks(np.arange(0.00, 0.21 + 1e-6, 0.05))  # Include 0.20
                ax.set_yticks(np.arange(-0.15, 0.11 + 1e-6, 0.05))  # Include 0.10

            # Set subplot properties
            greek_name = var_to_greek[var]
            ax.set_xlabel(
                f'{greek_name}',
                fontname='Times New Roman',
                fontsize=14,
                fontweight='bold'
            )
            ax.set_ylabel(
                f'ALE of {greek_name}',
                fontname='Times New Roman',
                fontsize=14
            )
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', which='both', width=1.5, length=6, direction='out')
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)

            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            # Add variable label above the subplot
            var_label = ax.text(
                0.5, 1.05, f"{greek_name} = ",
                transform=ax.transAxes,
                fontsize=13,
                ha='center',
                bbox=dict(
                    facecolor='#7ed4b3',
                    alpha=0.8,
                    edgecolor='black',
                    boxstyle='round,pad=0.2'
                )
            )
            self.var_labels[var] = var_label

        # Add K_dimless label placeholder in the bottom-left corner of the chart
        self.k_dimless_text = self.fig.text(
            0.02, 0.02, "K_dimless = ",
            fontsize=14,
            color='black',
            weight='bold',
            bbox=dict(facecolor='#64ffda', alpha=0.8, boxstyle='round,pad=0.5')
        )

        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------------------------- Event handling --------------------------
    def on_calculate(self):
        """Calculate button click event"""
        try:
            # Get input parameters
            params = {}
            for param_name in self.inputs:
                try:
                    value = float(self.inputs[param_name].get())
                    params[param_name] = value
                except ValueError:
                    messagebox.showerror("Input Error", f"{param_name} must be a number!")
                    return

            # Calculate intermediate variables
            intermediate = calculate_intermediate(params)
            mm_num, mm_den = calculate_MM(intermediate)
            k_dimless = calculate_K_dimless(mm_num, mm_den)

            # Update variable labels on subplots
            for var in user_order:
                if var in intermediate:
                    greek_name = var_to_greek[var]
                    current_val = intermediate[var]

                    # Update variable label above subplot
                    self.var_labels[var].set_text(f"{greek_name} = {current_val:.3f}")

                    # Update point marker on subplot
                    self.update_plot_point(var, intermediate[var])

            # Update K_dimless value in bottom-left corner
            self.k_dimless_text.set_text(f"K_dimless = {k_dimless:.2f}")

            # Refresh canvas
            self.canvas.draw()

        except ZeroDivisionError:
            messagebox.showerror("Error", "MM denominator cannot be zero!")
        except Exception as e:
            messagebox.showerror("Error", f"Unknown error: {str(e)}")

    def update_plot_point(self, var, current_val):
        """Update the point marker on the subplot"""
        ax = self.axes[var]

        # Remove existing red dot (if any), preserve ALE curve
        if hasattr(ax, 'marker_point') and ax.marker_point is not None:
            ax.marker_point.remove()
            ax.marker_point = None

        # Handle data (ensure x-axis is ascending)
        x_data, y_data = ale_data[var]
        if not np.all(np.diff(x_data) >= 0):
            sorted_indices = np.argsort(x_data)
            x_data = x_data[sorted_indices]
            y_data = y_data[sorted_indices]

        try:
            # Interpolate ALE value corresponding to current value
            ale_val = np.interp(current_val, x_data, y_data)
        except ValueError:
            return  # Skip if current value is out of ALE data range

        # Plot red dot and record
        ax.marker_point = ax.plot(current_val, ale_val, 'ro', markersize=6)[0]

    def on_default(self):
        """Default button click event: Restore default values"""
        for param_name, entry in self.inputs.items():
            # Clear current content
            entry.delete(0, tk.END)
            # Insert default value
            entry.insert(0, str(default_values[param_name]))

        # Automatically perform calculation
        self.on_calculate()

    def on_cancel(self):
        """Cancel button click event: Exit program"""
        self.on_close()

    def on_close(self):
        """Window close event"""
        plt.close(self.fig)
        self.root.destroy()
        sys.exit()


# -------------------------- Run GUI --------------------------
if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    root = tk.Tk()
    root.style = ttk.Style(root)
    root.style.theme_use("clam")

    # Set button style
    root.style.configure(
        'TButton',
        font=('Times New Roman', 15),
        background='#003366',
        foreground='white',
        borderwidth=1,
        padding=3
    )

    # Hover color
    root.style.map(
        'TButton',
        background=[('active', '#0055A4')],
        foreground=[('active', 'white')]
    )

    app = ALEGUI(root)
    root.mainloop()