"""
Oblig 6_Matematikk
Start: 01.04.21
Frist: 15.04.21, Kl_23:59

Oblig_tema:
    Kap: 9.2_Matriser
    Kap: 9.3_Determinanter
    Kap: 9.6_MinsteKvadratersMetode
    NewtonsMetode
    NumeriskIntegrasjon
    Feilestimat
    
Laget av: Sebastian Tveito Benjaminsen
"""

# Oppg 1) Inverterer 2x2 Matriser
"""
import numpy as np

a11 = float(input("tall for rad: 1, kollonne: 1, = "))
a12 = float(input("tall for rad: 1. kollonne: 2, = "))
a21 = float(input("tall for rad: 2, kollonne: 1, = "))
a22 = float(input("tall for rad: 2, kollonne: 2, = "))

def determinat(A_mat):
    A_det = A_mat[0][0]*A_mat[1][1] - A_mat[1][0]*A_mat[0][1]
    #print(A_det)
    return A_det

A_matrise = np.array([[a11, a12], [a21, a22]])
A_adjungert_matrise = np.array([[a22, -a12], [-a21, a11]])

#print(determinat(A_matrise))
#print("Determinaten =", determinat(A_matrise))


if determinat(A_matrise) != 0:
    print("Denne 2x2 matrisen kan inverteres")
    Invers_matrise = (1/determinat(A_matrise))*A_adjungert_matrise
    print("Inversen til A_matrise [A^-1]) = ", Invers_matrise)
    
    #Kontroll av inversen:
    def C11(A_mat, A_invers):
        C11 = A_invers[0][0]*A_mat[0][0] + A_invers[1][0]*A_mat[0][1]
        return C11
    
    def C12(A_mat, A_invers):
        C12 = A_invers[0][1]*A_mat[0][0] + A_invers[1][1]*A_mat[0][1]
        return C12

    def C21(A_mat, A_invers):
        C21 = A_invers[0][0]*A_mat[1][0] + A_invers[1][0]*A_mat[1][1]
        return C21

    def C22(A_mat, A_invers):
        C22 = A_invers[0][1]*A_mat[1][0] + A_invers[1][1]*A_mat[1][1]
        return C22
    
    C_matrise_tuple = ([C11(A_matrise, Invers_matrise), C12(A_matrise, Invers_matrise)], 
                       [C21(A_matrise, Invers_matrise), C22(A_matrise, Invers_matrise)])
    Kontroll_av_funn_av_InversMatrise = np.asarray(C_matrise_tuple)
    print("Dette er kontroll av inversen til matrisen = ", Kontroll_av_funn_av_InversMatrise)
else:
    print("Determinant = ", determinat(A_matrise), "Og kan derfor ikke inverteres")
"""

# Oppg 2) Determinerer 3x3 Matriser
"""

#Endre Vilkårlig matrise

import numpy as np

#Rad 1
a11 = float(input("tall for rad: 1, kollonne: 1, = "))
a12 = float(input("tall for rad: 1. kollonne: 2, = "))
a13 = float(input("tall for rad: 1, kollonne: 3, = "))

#Rad 2
a21 = float(input("tall for rad: 2, kollonne: 1, = "))
a22 = float(input("tall for rad: 2. kollonne: 2, = "))
a23 = float(input("tall for rad: 2, kollonne: 3, = "))

#Rad 3
a31 = float(input("tall for rad: 3, kollonne: 1, = "))
a32 = float(input("tall for rad: 3. kollonne: 2, = "))
a33 = float(input("tall for rad: 3, kollonne: 3, = "))

def determinat(A_mat):
    M11 = A_mat[1][1]*A_mat[2][2] - A_mat[2][1]*A_mat[1][2]
    M12 = A_mat[1][0]*A_mat[2][2] - A_mat[2][0]*A_mat[1][2]
    M13 = A_mat[1][0]*A_mat[2][1] - A_mat[2][0]*A_mat[1][1]
    a_det = A_mat[0][0]*M11 - A_mat[0][1]*M12 + A_mat[0][2]*M13
    #print(A_det)
    #print(M11)
    #print(M12)
    #print(M13)
    return a_det

A_matrise = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

print("Determinaten til 3x3 matrise er: ", determinat(A_matrise))
"""

# Oppg 3) Matrise multiplisering: 2x3 * 3x2
"""
import numpy as np
## 2x3 Matrise [Matrise A]
print("Matrise A:")
#Rad 1
a11 = float(input("tall for rad: 1, kollonne: 1, = "))
a12 = float(input("tall for rad: 1. kollonne: 2, = "))
a13 = float(input("tall for rad: 1, kollonne: 3, = "))

#Rad 2
a21 = float(input("tall for rad: 2, kollonne: 1, = "))
a22 = float(input("tall for rad: 2. kollonne: 2, = "))
a23 = float(input("tall for rad: 2, kollonne: 3, = "))

## 3x2 Matrise [Matrise B]
print("Matrise B:")

#Rad 1
b11 = float(input("tall for rad: 1, kollonne: 1, = "))
b12 = float(input("tall for rad: 1. kollonne: 2, = "))

#Rad 2
b21 = float(input("tall for rad: 2, kollonne: 1, = "))
b22 = float(input("tall for rad: 2. kollonne: 2, = "))

#Rad 3
b31 = float(input("tall for rad: 3, kollonne: 1, = "))
b32 = float(input("tall for rad: 3. kollonne: 2, = "))

def C11(A_mat, B_mat):
    C11 = B_mat[0][0]*A_mat[0][0] + B_mat[1][0]*A_mat[0][1] + B_mat[2][0]*A_mat[0][2]
    return C11

def C12(A_mat, B_mat):
    C12 = B_mat[0][1]*A_mat[0][0] + B_mat[1][1]*A_mat[0][1] + B_mat[2][1]*A_mat[0][2]
    return C12

def C21(A_mat, B_mat):
    C21 = B_mat[0][0]*A_mat[1][0] + B_mat[1][0]*A_mat[1][1] + B_mat[2][0]*A_mat[1][2]
    return C21

def C22(A_mat, B_mat):
    C22 = B_mat[0][1]*A_mat[1][0] + B_mat[1][1]*A_mat[1][1] + B_mat[2][1]*A_mat[1][2]
    return C22

A_matrise = np.array([[a11, a12, a13], [a21, a22, a23]])
B_matrise = np.array([[b11, b12], [b21, b22], [b31, b32]])

C_matrise_tuple = ([C11(A_matrise, B_matrise), C12(A_matrise, B_matrise)], [C21(A_matrise, B_matrise), C22(A_matrise, B_matrise)])
C_matrise = np.asarray(C_matrise_tuple)

print("Matrise A multiplisert med B [AB] = matrise C = ", C_matrise)
"""

# Oppg 4)


# Oppg 5)

import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(__file__)
txt_file = os.path.join(script_dir, "data_oppg5.txt")

M = np.loadtxt(txt_file)
x = M[:,0]
f = M[:,1]

plt.figure(3)
plt.plot(x, f, "r", linewidth = 4)
plt.show()

plt.figure(4)
plt.plot(x, f, "r*", linewidth = 4)
plt.show()
