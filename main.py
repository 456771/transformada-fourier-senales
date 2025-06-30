# Simulación de Transformada de Fourier en Python
import numpy as np
import matplotlib.pyplot as plt

# Configuración inicial
t = np.linspace(-1, 1, 1000)  # tiempo
dt = t[1] - t[0]              # resolución temporal
fs = 1 / dt                   # frecuencia de muestreo

# 1. Señal senoidal
f_seno = 5  # Hz
seno = np.sin(2 * np.pi * f_seno * t)

# 2. Pulso rectangular
pulso = np.where(np.abs(t) < 0.2, 1, 0)

# 3. Escalón unitario
escalon = np.where(t >= 0, 1, 0)

# Gráficas en el dominio del tiempo
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, seno)
plt.title("Señal senoidal")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, pulso)
plt.title("Pulso rectangular")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, escalon)
plt.title("Escalón unitario")
plt.grid()

plt.tight_layout()
plt.show()

# Función para calcular y graficar FFT
def graficar_fft(signal, t, titulo):
    N = len(signal)
    fft_result = np.fft.fft(signal)
    freq = np.fft.fftfreq(N, d=dt)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(freq[:N // 2], np.abs(fft_result)[:N // 2])
    plt.title(f"Magnitud de la FFT - {titulo}")
    plt.xlabel("Frecuencia [Hz]")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(freq[:N // 2], np.angle(fft_result)[:N // 2])
    plt.title(f"Fase de la FFT - {titulo}")
    plt.xlabel("Frecuencia [Hz]")
    plt.grid()
    plt.tight_layout()
    plt.show()

# FFT para cada señal
graficar_fft(seno, t, "Señal senoidal")
graficar_fft(pulso, t, "Pulso rectangular")
graficar_fft(escalon, t, "Escalón unitario")
