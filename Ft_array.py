from feature_scaling import *
import math

def ft_min(array):
    ret = False 
    for rows in array:
        ret = rows if ret is False or rows < ret else ret
    return ret

def ft_max(array):
    ret = False 
    for rows in array:
        ret = rows if ret is False or rows > ret else ret
    return ret

def ft_mean(array):
    return sum(array) / len(array)

def ft_std(array):
    lena = len(array) - 1
    suma = 0
    mean = ft_mean(array)
    for elem in array:
        suma += (elem - mean)**2
    std = (1 / lena) * suma
    std = math.sqrt(std)
    return std

def ft_count(array):
    return len(array)

def ft_percentile(array, percent):
   lena = len(array)
   array.sort() 
   n = lena * percent / 100
   return array[int(n)]

def ft_median(array):
    return ft_percentile(array.tolist(), 50)

def ft_first_quar(array):
    return ft_percentile(array.tolist(), 25)

def ft_third_quar(array):
    return ft_percentile(array.tolist(), 75)

def ft_normalize(array):
   return rescaling(array)

