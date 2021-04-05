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

def determinant(A_mat):
    A_det = A_mat[0][0]*A_mat[1][1] - A_mat[1][0]*A_mat[0][1]
    #print(A_det)
    return A_det

A_matrise = np.random.randint(-10, high=10, size=(2, 2), dtype='l')
A_adjungert_matrise = np.array([[A_matrise[1][1], - A_matrise[0][1]], [-A_matrise[1][0], A_matrise[0][0]]])


if determinant(A_matrise) != 0:
    print("Denne 2x2 matrisen kan inverteres")
    Invers_matrise = (1/determinant(A_matrise))*A_adjungert_matrise
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
    print("Determinant = ", determinant(A_matrise), "Og kan derfor ikke inverteres")
"""

# Oppg 2) Determinerer 3x3 Matriser
"""
import numpy as np

def determinat(A_mat):
    M11 = A_mat[1][1]*A_mat[2][2] - A_mat[2][1]*A_mat[1][2]
    M12 = A_mat[1][0]*A_mat[2][2] - A_mat[2][0]*A_mat[1][2]
    M13 = A_mat[1][0]*A_mat[2][1] - A_mat[2][0]*A_mat[1][1]
    a_det = A_mat[0][0]*M11 - A_mat[0][1]*M12 + A_mat[0][2]*M13
    #print(a_det)
    #print(M11)
    #print(M12)
    #print(M13)
    return a_det

Matrise_A = np.random.randint(-50, high=50, size=(3, 3), dtype='l')

print("Determinanten til 3x3 matrise er: ", determinant(Matrise_A))
"""
# Oppg 3) Matrise multiplisering: 2x3 * 3x2
"""
import numpy as np

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

# 2x3 Matrise
Matrise_A = np.random.randint(-10, high=10, size=(2, 3), dtype='l')
# 3x2 Matrise
Matrise_B = np.random.randint(-10, high=10, size=(3, 2), dtype='l')


C_matrise_tuple = ([C11(Matrise_A, Matrise_B), C12(Matrise_A, Matrise_B)], [C21(Matrise_A, Matrise_B), C22(Matrise_A, Matrise_B)])
C_matrise = np.asarray(C_matrise_tuple)

print("Matrise A multiplisert med B [AB] = matrise C = ", C_matrise)
"""
# Oppg 4)
"""
import sympy as sym

# Derivasjon:    
x = sym.Symbol("x")
funksjon = ((-x**3)/6) + (1/4)*sym.cos(2*x) + ((sym.E**x)/100) + (3/2)*x**2

funksjon_derivert = sym.diff(funksjon, x)
#print("Den deriverte til funksjonen = ", funksjon_derivert)

funksjon_dobbelderivert = sym.diff(funksjon_derivert, x)
#print("Den dobbel-deriverte til funksjonen = ", funksjon_dobbelderivert)

funksjon_trippelderivert = sym.diff(funksjon_dobbelderivert, x)
#print("Den trippel-deriverte til funksjonen = ", funksjon_trippelderivert)

funksjon_fjerdederivert = sym.diff(funksjon_trippelderivert, x)
#print("Den fjerde-deriverte til funksjonen = ", funksjon_fjerdederivert)


#Brukbare funksjoner i numpy
derivert_to = sym.lambdify(x, funksjon_dobbelderivert)
derivert_tre = sym.lambdify(x, funksjon_trippelderivert)
derivert_fire = sym.lambdify(x, funksjon_fjerdederivert)



def grense(x):
    # Legger inn funksjonen
    return (derivert_to(x)*(b-a)**3)/(24*n**2)
    
a = 0
b = 5/2
n = 20
m = 10
xh = 1

for i in range(m):
    xh = xh - derivert_tre(xh)/derivert_fire(xh) #Newtonsmetode
    #print(xh)

print("Ovregrense = ", grense(xh), "\n", "Feilestimeringen_midtpunktsmetoden = ", xh)
"""

# Oppg 5)
"""
import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(__file__)
txt_file = os.path.join(script_dir, "data_oppg5.txt")

M = np.loadtxt(txt_file)
x = M[:,0]
f = M[:,1]

plt.figure(1)
plt.plot(x, f, "r*", linewidth = 4)
plt.title("data_oppg5")
plt.xlabel("x-akse [x]")
plt.ylabel("y-akse [f(x)]")
plt.grid()
plt.show()

plt.figure(2)
plt.plot(x, f, "r", linewidth = 4)
plt.title("data_oppg5")
plt.xlabel("x-akse [x]")
plt.ylabel("y-akse [f(x)]")
plt.grid()
plt.show()

a = 0.0
b = 15.0
n = 46
dx = (b-a)/n
sum = 0
 
# Utregning av Areal:
#for i in range(0, n-1):
    #sum += f[i]    
    #sum += f[i+1]

#Tn = sum*dx/2

for i in range(0, n-1):
    sum += f[i]**2 + f[i]*f[i+1] + f[i+1]**2 #Et sigment

Volum = (1/3)*np.pi*dx*(sum) 
print("Volumet til funksjonen vist i graf: 'data_oppg5' er ",np.round(Volum, 2))
"""
# Oppg 6)
"""
import numpy as np
import matplotlib.pyplot as plt

a = np.array([1+2.7j, 3-4j, 5+6j, 1+0.5j, 0.3-4j, 5+1.6j, -1.1+2j, 5, 3.1-1.4j, 5+6j, -1-2j, 3j,])
x_akse = a.real
y_akse = a.imag

avstand_origo_punkt = list()
i = 0
    
for i in a:
  avstand = np.sqrt(i.imag**2 + i.real**2) 
  avstand_origo_punkt.append(avstand)

minste = np.min(avstand_origo_punkt)

index_nummer_for_minste_avstand_origo = avstand_origo_punkt.index(minste)

Kompleksetall_svar = a[index_nummer_for_minste_avstand_origo]

print("Det komplekse tallet som har kortest avstand til origo er ", Kompleksetall_svar, "\n", 
      "hvor lengden unna er origo er ", np.round(minste, 4),
      "og koordinaten til punktet", "x = ", x_akse[index_nummer_for_minste_avstand_origo],
      "y = ", y_akse[index_nummer_for_minste_avstand_origo])

#Graf med alle punktene:
plt.plot(x_akse, y_akse, "bo")  # Punktene til de komplekse-tallene
plt.plot(0, 0, "go")  # Origo
plt.title("Grafen som viser koordinater til de komplekse-tallene")
plt.xlabel("Reelle tall")
plt.ylabel("Imaginære tall")
plt.grid()
plt.show()

print("Komplekse tall - koordinater (x, y) ", "\n", np.column_stack((x_akse, y_akse)))
"""
