import numpy as np

# Fourier transform utilities. We seek for decomposition by exp(-iwt + ikx)

def fourier_transform(data, dts):
    f_data = np.fft.ifftshift(data)
    f_data = np.fft.fft2(f_data)
    f_data = np.fft.fftshift(f_data)

    # np.fft uses linear frequencies, so we multiply the arrays by (2 * np.pi)
    shape = data.shape
    w = np.fft.fftfreq(shape[0], d=dts) * (2 * np.pi)
    k = np.fft.fftfreq(shape[1], d=(2 * np.pi /  shape[1])) * (2 * np.pi)

    w = np.fft.fftshift(w)
    k = np.fft.fftshift(k)

    return f_data, w, k

def inverse_fourier_transform(f_data):
    data = np.fft.ifftshift(f_data)
    data = np.fft.ifft2(data)
    data = np.fft.fftshift(data)

    r_data = np.real(data)
    i_data = np.imag(data)

    return r_data, i_data
