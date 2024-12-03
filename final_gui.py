import tkinter as tk
from tkinter import messagebox
from circuit_class import Circuit
import customtkinter as ctk
import skrf as rf
from tkinter import filedialog, messagebox  # Importamos messagebox desde tkinter
import numpy as np

#Función que selecciona entre cargar o no cargar el archivo s2p
def toggle_options():
    global toggle_sel
    
    """Controla la visibilidad de los botones según los checkboxes activos."""
    if s2p_file_chckbx_si.get():
        s2p_file_chckbx_no.set(False)
        hide_info()   
        load_s2p_file()
        toggle_sel = 1
    elif s2p_file_chckbx_no.get():
        s2p_file_chckbx_si.set(False)  # Desactiva el otro checkbox
        hide_s2p()
        show_grl_info()
        toggle_sel = 0
    else:
        hide_info()     
        hide_s2p()
    #Ajustar tamaño automáticamente después de cambiar el contenido
    ventana.update_idletasks()

#Funcion para ocultar los datos de inf general
def hide_info():
    """Oculta las entradas adicionales."""
    init_freq_label.grid_remove()
    init_freq_entry.grid_remove()
    final_freq_label.grid_remove()
    final_freq_entry.grid_remove()
    steps_freq_label.grid_remove()
    steps_freq_entry.grid_remove()
    charac_imp_label.grid_remove()
    charac_imp_entry.grid_remove()

#Funcion para ocultar datos de archivo s2p
def hide_s2p():
    nodos_entry.grid_remove()
    nodos_label.grid_remove()
    
#Función para mostrar los datos de inf general    
def show_grl_info():
    init_freq_label.grid(row=3, column=0, sticky="w", pady=5)
    final_freq_label.grid(row=4, column=0, sticky="w", pady=5)
    steps_freq_label.grid(row=5, column=0, sticky="w", pady=5)
    charac_imp_label.grid(row=6, column=0, sticky="w", pady=5)
    component_number_entry.grid(row=1, column=1, padx=10, pady=5)
    input_nodes_entry.grid(row=2, column=1, padx=10, pady=5)
    init_freq_entry.grid(row=3, column=1, padx=10, pady=5)
    final_freq_entry.grid(row=4, column=1, padx=10, pady=5)
    steps_freq_entry.grid(row=5, column=1, padx=10, pady=5)
    charac_imp_entry.grid(row=6, column=1, padx=10, pady=5)
    # Botón para enviar la información general
    submit_gnl_inf_button = ctk.CTkButton(frame, text="Enviar Información", command=submit_grl_info)
    submit_gnl_inf_button.grid(row=7, column=0, columnspan=2, pady=10)

#Función para cargar archivo s2p
def load_s2p_file():
    
    global archivo_s2p
    
    # Usar filedialog para seleccionar el archivo
    archivo_s2p = filedialog.askopenfilename(filetypes=[("Archivos S2P", "*.s2p")])
    
    if archivo_s2p:
        messagebox.showinfo("Archivo Cargado", f"El archivo {archivo_s2p} se ha cargado correctamente.")
    else:
        messagebox.showerror("Error", "No se seleccionó ningún archivo.")
            
    if archivo_s2p:
        # Crear una entrada para ingresar los nodos entre los que está el archivo
        nodos_label.grid(row=0 + 3, column=0, columnspan=2, pady=10)
        nodos_entry.grid(row=1 + 4, column=0, columnspan=2, pady=10)
            
        # Función para procesar los nodos ingresados
        global nodos_lista
        nodos = nodos_entry.get()
        nodos_lista = nodos.split()
        if len(nodos_lista) == 3 and all(i.isdigit() for i in nodos_lista):
            messagebox.showinfo("Información", f"Archivo S2P: {archivo_s2p}\nNodos: {nodos_lista}")
        else:
            messagebox.showerror("Error", "Por favor, ingresa tres nodos numéricos válidos separados por espacio.")
    
    component_number_entry.grid(row=1, column=1, padx=10, pady=5)
    input_nodes_entry.grid(row=2, column=1, padx=10, pady=5)                
    # Botón para enviar la información general
    submit_gnl_inf_button = ctk.CTkButton(frame, text="Enviar Información", command=submit_grl_info)
    submit_gnl_inf_button.grid(row=7, column=0, columnspan=2, pady=10)

#Función para leer un archivo s2p
def read_s2p(filename):
    """
    Lee un archivo .s2p y extrae la frecuencia, los parámetros S y la resistencia de referencia.
    
    Args:
        filename (str): Ruta del archivo .s2p.
    
    Returns:
        tuple: 
            - freq (numpy.ndarray): Vector de frecuencias.
            - s_params (numpy.ndarray): Matriz de parámetros S (complejos).
            - z_ref (float): Resistencia de referencia.
    """
    frequencies = []
    s_parameters = []
    z_ref = None  # Inicializa la resistencia de referencia

    try:
        with open(filename, 'r') as file:
            for line in file:
                # Ignora comentarios y líneas vacías
                if line.startswith('!') or line.strip() == '':
                    continue
                
                # Procesa la línea de encabezado
                if line.startswith('#'):
                    header_parts = line.split()
                    try:
                        freq_type = (header_parts[1])
                        param_type = (header_parts[2])
                        result_type = (header_parts[3])
                        z_ref = float(header_parts[5])
                    except (IndexError, ValueError):
                        raise ValueError("El formato de la línea de encabezado no es válido.")
                    continue
                
                # Procesa los datos de los parámetros S
                parts = line.split()
                if len(parts) >= 9:
                    try:
                        freq = float(parts[0])
                        s11 = complex(float(parts[1]), float(parts[2]))
                        s21 = complex(float(parts[3]), float(parts[4]))
                        s12 = complex(float(parts[5]), float(parts[6]))
                        s22 = complex(float(parts[7]), float(parts[8]))
                    except ValueError:
                        raise ValueError("El formato de los datos de parámetros S no es válido.")
                    
                    #Matriz S para determinada frecuencia
                    matriz_s = np.array([[s11, s12], [s21, s22]], dtype=complex)
                    
                    #Guardar la frecuencia y la matriz
                    frequencies.append(freq)
                    s_parameters.append(matriz_s)

        # Verifica que se haya leído la resistencia de referencia
        if z_ref is None:
            raise ValueError("No se encontró la resistencia de referencia en el archivo.")
        
        steps = frequencies[1] - frequencies[0]
        return frequencies, steps, s_parameters, z_ref, freq_type, param_type, result_type
    
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo: {filename}")
    except Exception as e:
        raise RuntimeError(f"Ocurrió un error al procesar el archivo: {e}")

#Función para convertir parámetros S a ABCD
def s2ABCD(param, z_ref):
    
    if len(param) != 4:
        raise ValueError("Error")
    
    
    s11, s12, s21, s22 = param
    
    abcd_mat = [0]*4
    
    denom = 2 * s21
        
    abcd_mat[0] = complex(((1+s11) * (1-s22) + (s12*s21)) / (denom))
    abcd_mat[1] = complex(z_ref * ((1+s11) * (1+s22) - (s12*s21)) / (denom))
    abcd_mat[2] = complex((1/z_ref) * ((1+s11) * (1+s22) - (s12*s21)) / (denom)) 
    abcd_mat[3] = complex(((1-s11) * (1+s22) + (s12*s21)) / (denom))
    
    
    return abcd_mat

#Función para convertir parámetros ABCD a red de dos puertos    
def ABCD_2Port(param):
    
    if len(param) != 4:
        raise ValueError("Error")
    
    A, B, C, D = param
    
    Yc = complex(1 / B)
    Ya = complex((D/B) - 1)
    Yb = complex((A/B) - 1)
    
    Za = 1/Ya
    Zb = 1/Yb
    Zc = 1/Yc
    return Za, Zb, Zc
# Función para validar si una entrada es un número flotante
def es_numero(valor):
    try:
        float(valor)
        return True
    except ValueError:
        return False

# Función guarda la información general del circuito
def submit_grl_info():
    global box_list, i_freq, f_freq, s_freq, i_impedance, comp, input_nodes_list, ventana_2, s2p_freq, s2p_steps, s2p_params, s2p_z_ref, s2p_freq_type, s2p_param_type, s2p_result_type
        
    

    comp = nmb_comp.get()                    # guarda número de los componentes
    input_nodes = input_nodes_entry.get()    # guarda los nodos de entrada
    input_nodes_list = input_nodes.strip().split()  
    
    if toggle_sel == 0:
        i_freq = init_freq_entry.get()           # guarda la frecuencia inicial
        f_freq = final_freq_entry.get()          # guarda la frecuencia final
        s_freq = steps_freq_entry.get()          # guarda los pasos que se darán
        i_impedance = charac_imp_entry.get()     # guarda el valor de la impedancia característica
        print(comp)
    elif toggle_sel == 1:
        s2p_freq, s2p_steps, s2p_params, s2p_z_ref, s2p_freq_type, s2p_param_type, s2p_result_type = read_s2p(archivo_s2p)
        i_freq = s2p_freq[0]
        f_freq = s2p_freq[-1]
        s_freq = s2p_steps
        i_impedance = s2p_z_ref
    else:
        messagebox.showerror("ERROR", "Algo salió mal")
        
    try:
        input_nodes_list = [int(i) for i in input_nodes_list]
    except ValueError:
        messagebox.showerror("ERROR", "Los nodos de entrada deben ser números enteros, no más de un espacio entre ellos")
        return

    
    # Verificar si todos los campos están completos, si no manda un error
    if toggle_sel == 0:
        if not all([comp, input_nodes, i_freq, i_impedance, f_freq, s_freq]):
            messagebox.showwarning("ERROR", "Campos incompletos")
            return False
        # Validar que las frecuencias y la impedancia sean números válidos
        if not all([es_numero(i_freq), es_numero(f_freq), es_numero(s_freq), es_numero(i_impedance)]):
            messagebox.showwarning("ERROR", "Las entradas deben ser números enteros")
            return
    elif toggle_sel == 1:
        if not all([comp, input_nodes]):
            messagebox.showwarning("ERROR", "Campos incompletos!")
            print(input_nodes)
            return
        if not all([es_numero(i_freq), es_numero(f_freq), es_numero(s_freq), es_numero(i_impedance)]):
            messagebox.showwarning("ERROR", "Las entradas deben ser números enteros")
            return
    else:
        messagebox.showerror("ERROR", "Algo salió mal")   
    
    # Cerrar la ventana principal
    ventana.withdraw()

    # Nueva ventana para registro de componentes
    ventana_2 = ctk.CTk()
    ventana_2.title("Registro de Componentes y Lectura de Archivos")
    ventana_2.geometry("600x400")
    
    # Crear el frame para registro de componentes
    component_reg_frame = ctk.CTkFrame(ventana_2)
    component_reg_frame.pack(padx=20, pady=20, fill="both", expand=True)  # Usar pack en el marco principal

    # Encabezado del frame
    header_label = ctk.CTkLabel(component_reg_frame, text="Registro de Componentes", font=("Arial", 14, "bold"))
    header_label.grid(row=0, column=0, columnspan=2, pady=10)

    # Crear las etiquetas y cajas de texto de acuerdo al número de componentes registrados
    box_list = []  # Limpiar la lista de cajas antes de agregar nuevas
    for i in range(comp):
        label = ctk.CTkLabel(component_reg_frame, text=f"Componente {i+1}:")
        label.grid(row=i + 1, column=0, sticky="w", padx=5, pady=5)
        box = ctk.CTkEntry(component_reg_frame)
        box.grid(row=i + 1, column=1, padx=5, pady=5)
        box_list.append(box)

    # Botón para guardar la información de los componentes
    submit_comp_inf_button = ctk.CTkButton(component_reg_frame, text="Guardar Información", command=save_comp_inf)
    submit_comp_inf_button.grid(row=comp + 1, column=0, columnspan=2, pady=10)

    # Ejecutar la interfaz gráfica de la nueva ventana
    ventana_2.mainloop()

# Función para guardar la información de los componentes
def save_comp_inf():
    components_info = []

    # Procesar cada entrada y dividirla en partes
    for i in box_list:
        texto = i.get()
        partes = texto.split()

        # Verificar si la entrada tiene 4 partes
        if partes[0] in ['R', 'L', 'C', 'S', 'O', 'T']:
            if len(partes) == 4:
                components_info.append([partes[0], float(partes[1]), int(partes[2]), int(partes[3])])
            elif len(partes) == 3:
                components_info.append([partes[0], float(partes[1]), int(partes[2])])
            else:
                messagebox.showerror("ERROR", "Descripción del componente inválida, Ejemplo: C 1e4 1 2")
        else:
            messagebox.showerror("ERROR", "Describir el tipo de componente (R, L, C, S, O, T)")

    # Mostrar la información guardada
    if components_info:
        messagebox.showinfo("INFO", f"Se han guardado los siguientes componentes:\n{components_info}")

        circuit = Circuit(components=components_info,
                          input_nodes=input_nodes_list,
                          lower_freq_limit=float(i_freq),
                          upper_freq_limit=float(f_freq),
                          freq_step=float(s_freq),
                          z_charac=float(i_impedance)
                          )
        try:
            sim_circuit = circuit.run_simulation()
        except Exception as e:
            messagebox.showerror("ERROR", f"Ocurrió un error al simular el circuito: {e}")
        else:
            messagebox.showinfo("INFO", f"Se ha simulado el circuito correctamente")
            print(sim_circuit)
    else:
        messagebox.showwarning("Advertencia", "No se ha ingresado suficiente información.")

# Configuración de estilo de CustomTkinter
ctk.set_appearance_mode("System")  # Modo de apariencia: "Light", "Dark", "System"
ctk.set_default_color_theme("blue")  # Tema de color: "blue", "dark-blue", "green"

# Principal Window
ventana = ctk.CTk()
ventana.title("Simulador de Circuitos - Ing Microondas I")
ventana.geometry("800x400")
ventana.resizable(True, True)  # Habilitar redimensionamiento

# Crear el marco principal
frame = ctk.CTkFrame(ventana)
frame.pack(padx=20, pady=20, fill="both", expand=True)  # Usar pack en el marco principal

# Encabezado del frame
general_info_label = ctk.CTkLabel(frame, text="INFORMACIÓN GENERAL DEL CIRCUITO", font=("Arial", 14, "bold"))
general_info_label.grid(row=0, column=0, columnspan=2, pady=10)

# Etiquetas de información general
component_number_label = ctk.CTkLabel(frame, text="Número de componentes")
component_number_label.grid(row=1, column=0, sticky="w", pady=5)
nmb_comp = ctk.IntVar()
component_number_entry = ctk.CTkEntry(frame, textvariable=nmb_comp)
component_number_entry.grid(row=1, column=1, padx=10, pady=5)
s2p_file_chckbx_si = ctk.BooleanVar()
s2p_file_chckbx_no = ctk.BooleanVar()

input_nodes_label = ctk.CTkLabel(frame, text="Puertos de entrada")
input_nodes_label.grid(row=2, column=0, sticky="w", pady=5)
input_nodes_entry = ctk.CTkEntry(frame)
input_nodes_entry.grid(row=2, column=1, padx=10, pady=5)

s2p_file_label = ctk.CTkLabel(frame, text="¿Desea cargar archivo s2p?")
s2p_file_label.grid(row=1, column=3, sticky="w", pady=5)

# Crear el checkbox
chckbx_si = ctk.CTkCheckBox(frame, text="Sí", variable=s2p_file_chckbx_si, command=toggle_options)
chckbx_si.grid(row=1, column=4, sticky="w", pady=5)
chckbx_no = ctk.CTkCheckBox(frame, text="No", variable=s2p_file_chckbx_no, command=toggle_options)
chckbx_no.grid(row=1, column=5, sticky="w", pady=5)

#Crea las entradas solo se muestran si el usuario decide no usar un archivo s2p
init_freq_label = ctk.CTkLabel(frame, text="Frecuencia inicial (Hz)")
init_freq_entry = ctk.CTkEntry(frame)
final_freq_label = ctk.CTkLabel(frame, text="Frecuencia final (Hz)")
final_freq_entry = ctk.CTkEntry(frame)
steps_freq_label = ctk.CTkLabel(frame, text="Pasos de frecuencia (Hz)")
steps_freq_entry = ctk.CTkEntry(frame)
charac_imp_label = ctk.CTkLabel(frame, text="Valor impedancia característica (Ω)")
charac_imp_entry = ctk.CTkEntry(frame)

# Entradas para la información general
nmb_comp = ctk.IntVar()
component_number_entry = ctk.CTkEntry(frame, textvariable=nmb_comp)
input_nodes_entry = ctk.CTkEntry(frame)

#Crea las entradas usadas si el usuario decide usar archivo s2p
nodos_label = ctk.CTkLabel(frame, text="Ingresa los nodos (3 nodos separados por espacio):")
nodos_entry = ctk.CTkEntry(frame)


# Ejecutar la interfaz gráfica
ventana.mainloop()