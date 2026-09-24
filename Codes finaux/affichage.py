import numpy as np
import matplotlib.pyplot as plt

from pylab import *




def afficher(list_x,list_y,list_leg):
    #plt.ion()
    #plt.hold('on')
    plt.ion()
    for j,absic in enumerate(list_x):
        plt.plot(absic,list_y[j],'-',linewidth=2.0)
    plt.legend(list_leg)    
    plt.ioff()
    plt.show()
    
    
    
def afficherOndes(n,sol,listrho,listxi,ww):
    xxh = np.arange(0,1.0,0.05)   
    xx = np.array([])
    yyR = np.array([])
    yyI = np.array([])
    listxi.insert(0,listxi[0]-0.2)
    listxi.append(listxi[-1]+0.2)
    
    for i in range(n+2):
        #xxi = listpos[i] + xxh * (listpos[i+1] - listpos[i])
        xxi = np.arange(listxi[i],listxi[i+1],0.001)
        xx = np.concatenate((xx,xxi))
    
        Ai = sol[2*i]
        Bi = sol[2*i+1]
        
        
        rhoi = listrho[i]
    
        yyir = np.real(Ai*np.exp(1j*xxi*ww*np.sqrt(rhoi)) + Bi * np.exp(-1j*xxi*ww*np.sqrt(rhoi)))
        yyii = np.imag(Ai*np.exp(1j*xxi*ww*np.sqrt(rhoi)) + Bi * np.exp(-1j*xxi*ww*np.sqrt(rhoi)))
        yyR = np.concatenate((yyR,yyir))
        yyI = np.concatenate((yyI,yyii))
        
    plt.plot(xx,yyR,color='tab:blue',linewidth=3)
    plt.plot(xx,yyI,color='tab:orange',linewidth=3)
    
    plt.legend(['Partie réelle','Partie Imaginaire'])
    
    
    maxrho = np.max(listrho)
    minrho = np.min(listrho)
    if(maxrho == minrho):
        maxrho = minrho+1
    for i in range(n+2):    
        plt.fill_between(listxi[i:i+2],-2,2, facecolor='black', alpha=0.25*(listrho[i]-minrho+0.)/(maxrho-minrho+0.) )

    
    plt.ylim([-2,3])
    
    listxi.pop()
    listxi.pop(0)
    plt.show()    

def animOndesHarmo(n,sol,listrho,listxi,ww):
    dt = 0.01
    plt.ion()
    for t in np.arange(0,1,dt):
        plt.clf()
        coeffTime = np.exp(-1j*ww*t)
        afficherOndes(n,sol*coeffTime,listrho,listxi,ww)
        plt.pause(1e-3)

    
    plt.ioff()
    
    
def animBandeGap(n,sol,listrho,listxi,listww,listSol):
    plt.ion()
    for iw, ww in enumerate(listww):
        sol = listSol[iw]
        plt.clf()
        mytitle = 'Frequence : '+str('%03f' %ww)
        plt.title(mytitle)
        mystrR='|R|^2 = '+str('%01f' %np.abs(sol[1])**2 )
        plt.text(-0.2,2.5,mystrR)
        mystrT='|T|^2 = '+str('%01f' %np.abs(sol[2*n+2])**2)
        plt.text(-0.2,2.8,mystrT)
        mystrRpT='|R|^2 + |T|^2 = '+str('%01f' %(np.abs(sol[1])**2+np.abs(sol[2*n+2])**2))
        plt.text(-0.2,2.2,mystrRpT)
        
        afficherOndes(n,sol,listrho,listxi,ww)
        plt.pause(1e-3)
    
    plt.ioff()
    plt.clf()    