import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import customtkinter as ctk
from tkinter import ttk, messagebox


# Función para mostrar la matriz seleccionada
def mostrar():
    # Obtener el nombre de la matriz seleccionada
    matriz_seleccionada = combobox_matrices.get()

    if matriz_seleccionada in matrices:
        matriz_valores = matrices[matriz_seleccionada]
        mostrar_ventana_matriz(matriz_seleccionada, matriz_valores, freq)
    else:
        ctk.CTkMessagebox.show_info("Error", "Selecciona una matriz válida")

# Función para crear la ventana emergente que muestra la matriz
def mostrar_ventana_matriz(nombre, matriz, freqs):
    ventana_matriz = ctk.CTkToplevel()  # Crear una ventana secundaria
    ventana_matriz.title(f"Valores de {nombre}")
    ventana_matriz.geometry("300x200")
    
    # Crear un texto con la información de cada matriz para cada frecuencia
    texto = ""
    for i, matriz in enumerate(matriz):
        texto += f"Frecuencia: {freqs[i] / 1e6:.2f} MHz\n"
        texto += "\n".join(["\t".join(map(str, fila)) for fila in matriz])
        texto += "\n\n"
    
    # Mostrar el nombre de la matriz y sus valores
    etiqueta_nombre = ctk.CTkLabel(ventana_matriz, text=f"Matriz: {nombre}", font=("Arial", 14, "bold"))
    etiqueta_nombre.pack(pady=10)

    # Usar un cuadro de texto con scroll para mostrar matrices grandes
    text_box = ctk.CTkTextbox(ventana_matriz, wrap="none", height=300)
    text_box.insert("1.0", texto)
    text_box.configure(state="disabled")  # Hacerlo de solo lectura
    text_box.pack(padx=10, pady=10, fill="both", expand=True)

    # Botón para cerrar la ventana
    boton_cerrar = ctk.CTkButton(ventana_matriz, text="Cerrar", command=ventana_matriz.destroy)
    boton_cerrar.pack(pady=10)
    
#Función para convertir a dB
def parameter2dB(parameter):
    mag = np.abs(parameter)
    
    if mag == 0:
        return -np.inf
    #Conversión de magnitud a dB
    dB = 20 * np.log10(mag)
    
    return dB

#Función para convertir a Fase
def parameter2Phase(parameter):
    #Convierte el parámetro a fase en radianes
    phase = np.angle(parameter)
    
    #Convierte la fase en radianes a grados
    phase_degrees = np.degrees(phase)
    
    return phase_degrees

#Funcion para sacar parte real 
def parameter2real(parameter):
    real = parameter.real
    
    return real

#Funcion para sacar parte imaginaria 
def parameter2img(parameter):
    img = parameter.imag
    
    return img

def plot_mag(matriz_seleccionada, freq):
    if matriz_seleccionada in matrices:
        matriz_valores = matrices[matriz_seleccionada]
        n = matriz_valores[0].shape[0]  # Tamaño de la matriz

        fig, axs = plt.subplots(n, n, figsize=(12, 10))
        fig.suptitle('Magnitud vs Frecuencia', fontsize=16)

        step = (max(freq) - min(freq)) / 10
        ticks = np.arange(min(freq), max(freq) + step, step)

        for i in range(n):
            for j in range(n):
                mag = [np.abs(matriz[i, j]) for matriz in matriz_valores]
                axs[i, j].plot(freq, mag, marker='.', label=f'Mag({i+1},{j+1})')
                axs[i, j].set_title(f'Mag({i+1},{j+1})')
                axs[i, j].set_xlabel('Frecuencia (Hz)')
                axs[i, j].set_ylabel('Magnitud')
                axs[i, j].legend()
                axs[i, j].set_xlim([min(freq), max(freq)])
                axs[i, j].set_xticks(ticks)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    else:
        ctk.CTkMessagebox.show_info("Error", "Selecciona una matriz válida")

# Función para graficar fase de las posiciones de cada matriz nxn
def plot_Phase(matriz_seleccionada, freq):
    if matriz_seleccionada in matrices:
        matriz_valores = matrices[matriz_seleccionada]
        n = matriz_valores[0].shape[0]  # Tamaño de la matriz

        fig, axs = plt.subplots(n, n, figsize=(12, 10))
        fig.suptitle('Fase vs Frecuencia', fontsize=16)

        step = (max(freq) - min(freq)) / 10
        ticks = np.arange(min(freq), max(freq) + step, step)

        for i in range(n):
            for j in range(n):
                phase = [parameter2Phase(matriz[i, j]) for matriz in matriz_valores]
                axs[i, j].plot(freq, phase, marker='.', label=f'Fase({i+1},{j+1})')
                axs[i, j].set_title(f'Fase({i+1},{j+1})')
                axs[i, j].set_xlabel('Frecuencia (Hz)')
                axs[i, j].set_ylabel('Fase')
                axs[i, j].legend()
                axs[i, j].set_xlim([min(freq), max(freq)])
                axs[i, j].set_xticks(ticks)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    else:
        ctk.CTkMessagebox.show_info("Error", "Selecciona una matriz válida")

# Función para graficar dB de las posiciones de cada matriz nxn
def plot_dB(matriz_seleccionada, freq):
    if matriz_seleccionada in matrices:
        matriz_valores = matrices[matriz_seleccionada]
        n = matriz_valores[0].shape[0]  # Tamaño de la matriz

        fig, axs = plt.subplots(n, n, figsize=(12, 10))
        fig.suptitle('Decibeles vs Frecuencia', fontsize=16)

        step = (max(freq) - min(freq)) / 10
        ticks = np.arange(min(freq), max(freq) + step, step)

        for i in range(n):
            for j in range(n):
                dB = [parameter2dB(matriz[i, j]) for matriz in matriz_valores]
                print(dB)
                axs[i, j].plot(freq, dB, marker='.', label=f'dB({i+1},{j+1})')
                axs[i, j].set_title(f'dB({i+1},{j+1})')
                axs[i, j].set_xlabel('Frecuencia (Hz)')
                axs[i, j].set_ylabel('dB')
                axs[i, j].legend()
                axs[i, j].set_xlim([min(freq), max(freq)])
                axs[i, j].set_xticks(ticks)
                axs[i, j].set_xscale('log')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    else:
        ctk.CTkMessagebox.show_info("Error", "Selecciona una matriz válida")

# Función para graficar parte real de las posiciones de cada matriz nxn
def plot_real(matriz_seleccionada, freq):
    if matriz_seleccionada in matrices:
        matriz_valores = matrices[matriz_seleccionada]
        n = matriz_valores[0].shape[0]  # Tamaño de la matriz

        fig, axs = plt.subplots(n, n, figsize=(12, 10))
        fig.suptitle('Real vs Frecuencia', fontsize=16)

        step = (max(freq) - min(freq)) / 10
        ticks = np.arange(min(freq), max(freq) + step, step)

        for i in range(n):
            for j in range(n):
                real = [parameter2real(matriz[i, j]) for matriz in matriz_valores]
                axs[i, j].plot(freq, real, marker='.', label=f'Re({i+1},{j+1})')
                axs[i, j].set_title(f'Re({i+1},{j+1})')
                axs[i, j].set_xlabel('Frecuencia (Hz)')
                axs[i, j].set_ylabel('Real')
                axs[i, j].legend()
                axs[i, j].set_xlim([min(freq), max(freq)])
                axs[i, j].set_xticks(ticks)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    else:
        ctk.CTkMessagebox.show_info("Error", "Selecciona una matriz válida")

# Función para graficar parte imaginaria de las posiciones de cada matriz nxn
def plot_img(matriz_seleccionada, freq):
    if matriz_seleccionada in matrices:
        matriz_valores = matrices[matriz_seleccionada]
        n = matriz_valores[0].shape[0]  # Tamaño de la matriz

        fig, axs = plt.subplots(n, n, figsize=(12, 10))
        fig.suptitle('Imaginario vs Frecuencia', fontsize=16)

        step = (max(freq) - min(freq)) / 10
        ticks = np.arange(min(freq), max(freq) + step, step)

        for i in range(n):
            for j in range(n):
                img = [parameter2img(matriz[i, j]) for matriz in matriz_valores]
                axs[i, j].plot(freq, img, marker='.', label=f'Im({i+1},{j+1})')
                axs[i, j].set_title(f'Im({i+1},{j+1})')
                axs[i, j].set_xlabel('Frecuencia (Hz)')
                axs[i, j].set_ylabel('Imaginario')
                axs[i, j].legend()
                axs[i, j].set_xlim([min(freq), max(freq)])
                axs[i, j].set_xticks(ticks)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    else:
        ctk.CTkMessagebox.show_info("Error", "Selecciona una matriz válida")

def plot_smith(matriz_seleccionada, freq):
    if matriz_seleccionada in matrices:
        matriz_valores = matrices[matriz_seleccionada]
        n = matriz_valores[0].shape[0]  # Tamaño de la matriz

        fig, axs = plt.subplots(n, n, figsize=(12, 12))
        fig.suptitle('Carta de Smith', fontsize=16)

        for i in range(n):
            for j in range(n):
                s_param = [matriz[i, j] for matriz in matriz_valores]
                frequency = rf.Frequency.from_f(freq, unit='hz')
                network = rf.Network(frequency=frequency, s=s_param)
                network.plot_s_smith(m=i, n=j, ax=axs[i, j], label=f'S{i+1}{j+1}')
                axs[i, j].set_title(f'S{i+1}{j+1}')
                axs[i, j].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    else:
        ctk.CTkMessagebox.show_info("Error", "Selecciona una matriz válida")


def plot_polar(matriz_seleccionada):
    if matriz_seleccionada in matrices:
        matriz_valores = matrices[matriz_seleccionada]
        n = matriz_valores[0].shape[0]  # Tamaño de la matriz

        fig, axs = plt.subplots(n, n, subplot_kw={'projection': 'polar'}, figsize=(12, 12))
        fig.suptitle('Gráfica Polar', fontsize=16)

        for i in range(n):
            for j in range(n):
                mag = [np.abs(matriz[i, j]) for matriz in matriz_valores]
                phase = [np.angle(matriz[i, j]) for matriz in matriz_valores]
                axs[i, j].plot(phase, mag, marker='.', label=f'Polar({i+1},{j+1})')
                axs[i, j].set_title(f'Polar({i+1},{j+1})')
                axs[i, j].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    else:
        ctk.CTkMessagebox.show_info("Error", "Selecciona una matriz válida")
#Función selectora para graficar
def graficar():
    try:
        # Obtener la función seleccionada
        sel_mat_i = combobox_matrices.get()
        sel_plot_i = sel_plot.get()
        
        if sel_mat_i != "Archivo S2P":
            freq_plot = frecuencias
        elif sel_mat_i == "Archivo S2P":
            freq_plot = s2p_freq
            
        # Elegir la función según la selección del usuario
        if sel_mat_i == "Z":
            if sel_plot_i == "Magnitud vs Frecuencia":
                plot_mag(sel_mat_i, freq_plot)
            elif sel_plot_i == "Fase vs Frecuencia":
                plot_Phase(sel_mat_i, freq_plot) 
            elif sel_plot_i == "dB vs Frecuencia":
                plot_dB(sel_mat_i, freq_plot)
            elif sel_plot_i == "Real vs Frecuencia":
                plot_real(sel_mat_i, freq_plot)
            elif sel_plot_i == "Imaginario vs Frecuencia":
                plot_img(sel_mat_i, freq_plot)
            elif sel_plot_i == "Gráfica Polar":
                plot_polar(sel_mat_i, freq_plot)
            elif sel_plot_i == "Carta de Smith":
                messagebox.showerror("Error", "Selecciona una función válida.")
                return
            else:
                messagebox.showerror("Error", "Selecciona una función válida.")
                return
        elif sel_mat_i == "Y":
            if sel_plot_i == "Magnitud vs Frecuencia":
                plot_mag(sel_mat_i, freq_plot)
            elif sel_plot_i == "Fase vs Frecuencia":
                plot_Phase(sel_mat_i, freq_plot) 
            elif sel_plot_i == "db vs Frecuencia":
                plot_dB(sel_mat_i, freq_plot)
            elif sel_plot_i == "Real vs Frecuencia":
                plot_real(sel_mat_i, freq_plot)
            elif sel_plot_i == "Imaginario vs Frecuencia":
                plot_img(sel_mat_i, freq_plot)
            elif sel_plot_i == "Gráfica Polar":
                plot_polar(sel_mat_i, freq_plot)
            elif sel_plot_i == "Carta de Smith":
                messagebox.showerror("Error", "Selecciona una función válida.")
                return
            else:
                messagebox.showerror("Error", "Selecciona una función válida.")
                return
        elif sel_mat_i == "ABCD":
            if sel_plot_i == "Magnitud vs Frecuencia":
                plot_mag(sel_mat_i, freq_plot)
            elif sel_plot_i == "Fase vs Frecuencia":
                plot_Phase(sel_mat_i, freq_plot) 
            elif sel_plot_i == "db vs Frecuencia":
                plot_dB(sel_mat_i, freq_plot)
            elif sel_plot_i == "Real vs Frecuencia":
                plot_real(sel_mat_i, freq_plot)
            elif sel_plot_i == "Imaginario vs Frecuencia":
                plot_img(sel_mat_i, freq_plot)
            elif sel_plot_i == "Gráfica Polar":
                plot_polar(sel_mat_i, freq_plot)
            elif sel_plot_i == "Carta de Smith":
                messagebox.showerror("Error", "Selecciona una función válida.")
                return
            else:
                messagebox.showerror("Error", "Selecciona una función válida.")
                return
        elif sel_mat_i == "S":
            if sel_plot_i == "Magnitud vs Frecuencia":
                plot_mag(sel_mat_i, freq_plot)
            elif sel_plot_i == "Fase vs Frecuencia":
                plot_Phase(sel_mat_i, freq_plot) 
            elif sel_plot_i == "db vs Frecuencia":
                plot_dB(sel_mat_i, freq_plot)
            elif sel_plot_i == "Real vs Frecuencia":
                plot_real(sel_mat_i, freq_plot)
            elif sel_plot_i == "Imaginario vs Frecuencia":
                plot_img(sel_mat_i, freq_plot)
            elif sel_plot_i == "Gráfica Polar":
                plot_polar(sel_mat_i, freq_plot)
            elif sel_plot_i == "Carta de Smith":
                plot_smith(sel_mat_i, freq_plot)
                return
            else:
                messagebox.showerror("Error", "Selecciona una función válida.")
                return               
        else:
            messagebox.showerror("Error", "Selecciona una función válida.")
            return    
    except Exception as e:
        messagebox.showerror("Error", f"Hubo un error al graficar: {e}")


# Lista de frecuencias
#frecuencias = [1e6, 5e6, 10e6, 50e6, 100e6]  # Frecuencias en Hz

# Diccionario de matrices dependientes de frecuencia
#matrices = {
#    "Z": [np.array([[1 + 1j * f, 2 + 0.5j * f], [3 - 0.5j * f, 4 + 1j * f]]) for f in frecuencias],
#    "Y": [np.array([[0.5 - 0.2j * f, 1.5 + 0.3j * f], [-0.5 + 0.7j * f, 0.7 - 0.4j * f]]) for f in frecuencias],
#    "ABCD": [np.array([[2 + 0.5j * f, 1 - 0.3j * f], [-1 + 0.2j * f, 3 - 0.6j * f]]) for f in frecuencias],
#    "S": [np.array([[2 + 0.5j * f, 1 - 0.3j * f], [-1 + 0.2j * f, 3 - 0.6j * f]]) for f in frecuencias],
#}

def Vizualizer(matrix, f): 
    global s2p_params, s2p_freq, z_ref, combobox_matrices, sel_plot, boton_graficar, boton_mostrar, ventana_4, freq, matrices, frecuencias

    matrices = matrix
    freq = f
    frecuencias = f
    #Archivo s2p

    #Agregar archivo s2p al diccionario de matrices
    #matrices['Archivo S2P'] = s2p_params

    # Configuración de customtkinter
    ctk.set_appearance_mode("Dark")  # Modo de apariencia: "Light", "Dark", "System"
    ctk.set_default_color_theme("dark-blue")  # Tema de color: "blue", "green", "dark-blue"

    # Crear la ventana principal
    ventana_4 = ctk.CTk()
    ventana_4.title("Resultados")
    ventana_4.geometry("400x300")

    # Etiqueta informativa
    etiqueta = ctk.CTkLabel(ventana_4, text="Selecciona una matriz y la acción que desea realizar. Si desea graficar, seleccione también el tipo de gráfica")
    etiqueta.pack(pady=10)

    # Combobox para seleccionar la matriz
    combobox_matrices = ctk.CTkComboBox(ventana_4, values=list(matrices.keys()))
    combobox_matrices.set(list(matrices.keys())[0])  # Seleccionar la primera opción por defecto
    combobox_matrices.pack(pady=10)

    # Crear un combobox (selector) para elegir el tipo de gráfica
    sel_plot = ctk.CTkComboBox(
        ventana_4,
        values=["Magnitud vs Frecuencia", "Fase vs Frecuencia", 
                "dB vs Frecuencia", "Real vs Frecuencia", 
                "Imaginario vs Frecuencia", "Gráfica Polar", "Carta de Smith"]
    )
    sel_plot.set("Magnitud vs Frecuencia")  # Valor predeterminado
    sel_plot.pack(pady=10)

    # Botón para graficar
    boton_graficar = ctk.CTkButton(ventana_4, text="Graficar", command=graficar)
    boton_graficar.pack(pady=20)

    # Botón para mostrar la matriz seleccionada
    boton_mostrar = ctk.CTkButton(ventana_4, text="Mostrar Matriz", command=mostrar)
    boton_mostrar.pack(pady=20)

    # Iniciar la GUI
    ventana_4.mainloop()

