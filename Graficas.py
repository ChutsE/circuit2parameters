import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import customtkinter as ctk
from tkinter import messagebox, filedialog


#Funcion que convierte de RI a MA
def ri2ma(complejos):
    """
    Convierte números complejos a magnitud y ángulo en grados.
    
    :param complejos: Número complejo, lista o array de números complejos.
    :return: Tupla (magnitudes, ángulos) donde:
             - magnitudes: Magnitud de los números complejos.
             - ángulos: Ángulo en grados de los números complejos.
    """
    # Calcular magnitud y ángulo
    magnitudes = np.abs(complejos)
    angulos = np.angle(complejos, deg=True)
    
    resultado = [(m, a) for m, a in zip(magnitudes, angulos)]

    return resultado

# Función para mostrar la matriz seleccionada
def mostrar():
    # Obtener el nombre de la matriz seleccionada
    matriz_seleccionada = combobox_matrices.get()
    opcion_conversion = box_mostrar.get()
    opcion_conversion = box_mostrar.get()

    if matriz_seleccionada in matrices:
        matriz_valores = matrices[matriz_seleccionada]
        
        if opcion_conversion == "MA":
            matriz_valores = ri2ma(matriz_valores)
        elif opcion_conversion == "RI":
            matriz_valores = matriz_valores
        else:
            messagebox.showinfo("Error", "Selecciona una opción válida")
            return
        mostrar_ventana_matriz(matriz_seleccionada, matriz_valores, frecuencias)
    else:
        messagebox.showinfo("Error", "Selecciona una matriz válida")

#Funcion para seleccionar entre leer y graficar
def toggle_display():
    global toggle_dis   
    """Controla la visibilidad de los botones según los checkboxes activos."""
    if i_chckbx_read.get():
        i_chckbx_plot.set(False)
        hide_plot()   
        show_read()
        toggle_dis = 1
    elif i_chckbx_plot.get():
        i_chckbx_read.set(False)  # Desactiva el otro checkbox
        hide_read()
        show_plot()
        toggle_dis = 0
        
        if opcion_conversion == "MA":
            matriz_valores = ri2ma(matriz_valores)
        elif opcion_conversion == "RI":
            matriz_valores = matriz_valores
        else:
            messagebox.showinfo("Error", "Selecciona una opción válida")
            return
        mostrar_ventana_matriz(matriz_seleccionada, matriz_valores, frecuencias)
    else:
        messagebox.showinfo("Error", "Selecciona una matriz válida")

#Funcion para seleccionar entre leer y graficar
def toggle_display():
    global toggle_dis   
    """Controla la visibilidad de los botones según los checkboxes activos."""
    if i_chckbx_read.get():
        i_chckbx_plot.set(False)
        hide_plot()   
        show_read()
        toggle_dis = 1
    elif i_chckbx_plot.get():
        i_chckbx_read.set(False)  # Desactiva el otro checkbox
        hide_read()
        show_plot()
        toggle_dis = 0
    else:
        hide_plot()     
        hide_read()
    #Ajustar tamaño automáticamente después de cambiar el contenido
    ventana_4.update_idletasks()

#Función para mostrar botones de lectura
def show_read():
    box_mostrar.grid(row=3, column=0, padx=10, pady=10)
    boton_mostrar.grid(row=4, column=0, padx=10, pady=10)
    
#Función para ocultar botones de lectura
def hide_read():
    box_mostrar.grid_forget()
    boton_mostrar.grid_forget()
    
#Función para mostrar botones de graficación
def show_plot():
    sel_plot.grid(row=3, column=0, padx=10, pady=10)
    boton_graficar.grid(row=4, column=0, padx=10, pady=10)

#Función para ocultar botones de graficación
def hide_plot():
    sel_plot.grid_forget()
    boton_graficar.grid_forget()
    

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
                axs[i, j].plot(freq, dB, marker='.', label=f'dB({i+1},{j+1})')
                axs[i, j].set_title(f'dB({i+1},{j+1})')
                axs[i, j].set_xlabel('Frecuencia (Hz)')
                axs[i, j].set_ylabel('dB')
                axs[i, j].legend()
                axs[i, j].set_xlim([min(freq), max(freq)])
                axs[i, j].set_xticks(ticks)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    else:
        ctk.CTkMessagebox.show_info("Error", "Selecciona una matriz válida")

def plot_dB_vs_dB(matriz_seleccionada, freq):
    if matriz_seleccionada in matrices:
        matriz_valores = matrices[matriz_seleccionada]
        n = matriz_valores[0].shape[0]  # Tamaño de la matriz

        fig, axs = plt.subplots(n, n, figsize=(12, 10))
        fig.suptitle('dB vs dB', fontsize=16)

        step = (max(freq) - min(freq)) / 10
        ticks = np.arange(min(freq), max(freq) + step, step)

        for i in range(n):
            for j in range(n):
                dB = [parameter2dB(matriz[i, j]) for matriz in matriz_valores]
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
                s_param = np.array([matriz[i, j] for matriz in matriz_valores])
                s_param = s_param.reshape((len(freq), 1, 1))  # Reshape to (frequency_points, 1, 1)
                frequency = rf.Frequency.from_f(freq, unit='hz')
                network = rf.Network(frequency=frequency, s=s_param)
                network.plot_s_smith(m=0, n=0, ax=axs[i, j], label=f'S{i+1}{j+1}')
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
        freq_plot = frecuencias
            
        # Elegir la función según la selección del usuario
        if sel_mat_i == "Z":
            if sel_plot_i == "Magnitud vs Frecuencia":
                plot_mag(sel_mat_i, freq_plot)
            elif sel_plot_i == "Fase vs Frecuencia":
                plot_Phase(sel_mat_i, freq_plot) 
            elif sel_plot_i == "dB vs Frecuencia":
                plot_dB(sel_mat_i, freq_plot)
            elif sel_plot_i == "dB vs dB":
                plot_dB_vs_dB(sel_mat_i, freq_plot)
            elif sel_plot_i == "Real vs Frecuencia":
                plot_real(sel_mat_i, freq_plot)
            elif sel_plot_i == "Imaginario vs Frecuencia":
                plot_img(sel_mat_i, freq_plot)
            elif sel_plot_i == "Gráfica Polar":
                plot_polar(sel_mat_i)
            else:
                messagebox.showerror("Error", "Selecciona una función válida.")
                return
        elif sel_mat_i == "Y":
            if sel_plot_i == "Magnitud vs Frecuencia":
                plot_mag(sel_mat_i, freq_plot)
            elif sel_plot_i == "Fase vs Frecuencia":
                plot_Phase(sel_mat_i, freq_plot) 
            elif sel_plot_i == "dB vs Frecuencia":
                plot_dB(sel_mat_i, freq_plot)
            elif sel_plot_i == "dB vs dB":
                plot_dB_vs_dB(sel_mat_i, freq_plot)
            elif sel_plot_i == "Real vs Frecuencia":
                plot_real(sel_mat_i, freq_plot)
            elif sel_plot_i == "Imaginario vs Frecuencia":
                plot_img(sel_mat_i, freq_plot)
            elif sel_plot_i == "Gráfica Polar":
                plot_polar(sel_mat_i)
            else:
                messagebox.showerror("Error", "Selecciona una función válida.")
                return
        elif sel_mat_i == "ABCD":
            if sel_plot_i == "Magnitud vs Frecuencia":
                plot_mag(sel_mat_i, freq_plot)
            elif sel_plot_i == "Fase vs Frecuencia":
                plot_Phase(sel_mat_i, freq_plot) 
            elif sel_plot_i == "dB vs Frecuencia":
                plot_dB(sel_mat_i, freq_plot)
            elif sel_plot_i == "dB vs dB":
                plot_dB_vs_dB(sel_mat_i, freq_plot)
            elif sel_plot_i == "Real vs Frecuencia":
                plot_real(sel_mat_i, freq_plot)
            elif sel_plot_i == "Imaginario vs Frecuencia":
                plot_img(sel_mat_i, freq_plot)
            elif sel_plot_i == "Gráfica Polar":
                plot_polar(sel_mat_i)
            else:
                messagebox.showerror("Error", "Selecciona una función válida.")
                return
        elif sel_mat_i == "S":
            if sel_plot_i == "Magnitud vs Frecuencia":
                plot_mag(sel_mat_i, freq_plot)
            elif sel_plot_i == "Fase vs Frecuencia":
                plot_Phase(sel_mat_i, freq_plot) 
            elif sel_plot_i == "dB vs Frecuencia":
                plot_dB(sel_mat_i, freq_plot)
            elif sel_plot_i == "dB vs dB":
                plot_dB_vs_dB(sel_mat_i, freq_plot)
            elif sel_plot_i == "Real vs Frecuencia":
                plot_real(sel_mat_i, freq_plot)
            elif sel_plot_i == "Imaginario vs Frecuencia":
                plot_img(sel_mat_i, freq_plot)
            elif sel_plot_i == "Gráfica Polar":
                plot_polar(sel_mat_i)
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

#Funcion para guardar archivo s2p
def guardar():
    # Crear una nueva ventana
    ventana_guardar = ctk.CTkToplevel(ventana_4)
    ventana_guardar.title("Guardar .s2p como")
    ventana_guardar.geometry("300x150")

    # Etiqueta para el campo de entrada
    label = ctk.CTkLabel(ventana_guardar, text="Nombre del archivo:")
    label.grid(row=0, column=0, pady=10)

    # Campo de entrada para el nombre del archivo
    nombre_entry = ctk.CTkEntry(ventana_guardar, placeholder_text="Escribe el nombre del archivo")
    nombre_entry.grid(row=0, column=1,pady=10, padx=20)
    # Botón Descargar
    descargar_boton = ctk.CTkButton(
        ventana_guardar,
        text="Descargar",
        command=lambda: download_s2p(matrices, frecuencias, nombre_entry.get()),
        )
    descargar_boton.grid(row=1, column=0, pady=10)

#Función para crear archivo s2p
def s2p_file(matrices, frecuencias, filename):
    if "S" not in matrices:
        raise ValueError("No se encontró la matriz S en el diccionario.")
    try:
        with open(filename, "w") as file:
            # Escribir la resistencia de referencia
            file.write(f"! S2P File: Measurements: S-parameters\n")
            file.write(f"# Hz S RI R {int(z_ref)}\n")  # Parámetros S, formato RI, resistencia de referencia
            
            # Escribir los parámetros S
            for idx, frecuencia in enumerate(frecuencias):
                s = matrices["S"][idx]
                file.write(f"{frecuencia}")
                for i in range(s.shape[0]):
                    for j in range(s.shape[1]):
                        file.write(f" {s[i, j].real} {s[i, j].imag}")
                file.write("\n")
            messagebox.showinfo("Archivo Guardado", f"Se ha guardado el archivo: {filename}")
        return True
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error al guardar el archivo: {e}")
        return False
    
#Función para descargar archivo s2p
def download_s2p(matrices, frecuencias, filename):
    if not filename:
        messagebox.showerror("Error", "El nombre del archivo no puede estar vacío.")
        return
    
    file = filedialog.asksaveasfilename(initialfile=f"{filename}.s2p", defaultextension=".s2p", filetypes=[("S2P Files", "*.s2p"), ("Todos los archivos", "*.*")])
    
    if file:
        s2p_file(matrices, frecuencias, file)


def Vizualizer(matrix, f, z0): 
    global box_mostrar, i_chckbx_plot, i_chckbx_read, combobox_matrices, sel_plot, boton_graficar, boton_mostrar, ventana_4, matrices, frecuencias, z_ref

    matrices = matrix
    frecuencias = f
    z_ref = z0


# Configuración de customtkinter
ctk.set_appearance_mode("Dark")  # Modo de apariencia: "Light", "Dark", "System"
ctk.set_default_color_theme("dark-blue")  # Tema de color: "blue", "green", "dark-blue"

    # Crear la ventana principal
    ventana_4 = ctk.CTk()
    ventana_4.title("Resultados")
    ventana_4.geometry("500x250")

    # Etiqueta informativa
    etiqueta = ctk.CTkLabel(ventana_4, text="Selecciona una matriz y la acción que desea realizar.")
    etiqueta.grid(row=0, column=0, columnspan=6, pady=10)
    i_chckbx_read = ctk.BooleanVar()
    i_chckbx_plot = ctk.BooleanVar()

    # Combobox para seleccionar la matriz
    combobox_matrices = ctk.CTkComboBox(ventana_4, values=list(matrices.keys()))
    combobox_matrices.set(list(matrices.keys())[0])  # Seleccionar la primera opción por defecto
    combobox_matrices.grid(row=1, column=0, columnspan=3, sticky="w", pady=5)

    # Crear el checkbox
    chckbx_read = ctk.CTkCheckBox(ventana_4, text="Ver Parámetros", variable=i_chckbx_read, command=toggle_display)
    chckbx_read.grid(row=1, column=4, sticky="w", pady=5)
    chckbx_plot = ctk.CTkCheckBox(ventana_4, text="Graficar", variable=i_chckbx_plot, command=toggle_display)
    chckbx_plot.grid(row=1, column=5, sticky="w", pady=5)

    #Botón para guardar archivo s2p
    boton_guardar = ctk.CTkButton(ventana_4, text="Guardar .s2p", command=guardar)
    boton_guardar.grid(row=1, column=6, sticky="w", pady=5)

    # Crear un combobox (selector) para elegir el tipo de gráfica
    sel_plot = ctk.CTkComboBox(
        ventana_4,
        values=["Magnitud vs Frecuencia", "Fase vs Frecuencia", 
                "dB vs Frecuencia", "dB vs dB", "Real vs Frecuencia",
                "Imaginario vs Frecuencia", "Gráfica Polar", "Carta de Smith"]
    )
    sel_plot.set("Magnitud vs Frecuencia")  # Valor predeterminado

    # Botón para graficar
    boton_graficar = ctk.CTkButton(ventana_4, text="Graficar", command=graficar)

    # Botón para mostrar la matriz seleccionada
    box_mostrar = ctk.CTkComboBox(ventana_4, values=["RI","MA"])
    boton_mostrar = ctk.CTkButton(ventana_4, text="Mostrar Matriz", command=mostrar)

    ventana_4.mainloop()

