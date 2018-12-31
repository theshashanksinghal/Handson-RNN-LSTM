import numpy as np

def softmax(x):
    ex=np.exp(x-np.max(x))
    return ex/ex.sum(axis=0)

def sigmoid(x):
    return 1/(1+np.exp(-x))
	
							 # RNN FORWARD PROP #

def rnn_cell(xt,a_prev,parameters):
    Waa=parameters["Waa"]
    Wax=parameters["Wax"]
    Wya=parameters["Wya"]
    ba=parameters["ba"]
    by=parameters["by"]
    
    a_next=np.tanh(np.dot(Waa,a_prev)+np.dot(Wax,xt)+ba)
    y_pred=softmax(np.dot(Wya,a_next)+by)
    cache=(a_next,a_prev,xt,parameters)
    return a_next,y_pred,cache

##########################################################
np.random.seed(1)
xt=np.random.randn(3,10)
a_prev=np.random.randn(5,10)
Waa=np.random.randn(5,5)
Wax=np.random.randn(5,3)
Wya=np.random.randn(2,5)
ba=np.random.randn(5,1)
by=np.random.randn(2,1)
parameters={"Waa":Waa,"Wax":Wax,"Wya":Wya,"ba":ba,"by":by}
a_next,y_pred,cache=rnn_cell(xt,a_prev,parameters)
print("a_next[4]=",a_next[4])
print("a_shape",a_next.shape)

############################################################

def rnn_forward(x,a0,parameters):
    caches=[]
    n_x,m,T_x=x.shape
    a_next=a0
    n_y,n_a=parameters["Wya"].shape
    a=np.zeros((n_a,m,T_x))
    y_predt=np.zeros((n_y,m,T_x))
    for t in range(T_x):
        a_next,y_pred,cache=rnn_cell(x[:,:,t],a_next,parameters)
        a[:,:,t]=a_next
        y_predt[:,:,t]=y_pred
        caches.append(cache)
    caches=(caches,x)
    return a,y_predt,caches

###############################################################
np.random.seed(1)
x=np.random.randn(3,10,4)
a0=np.random.randn(5,10)
Waa=np.random.randn(5,5)
Wax=np.random.randn(5,3)
Wya=np.random.randn(2,5)
ba=np.random.randn(5,1)
by=np.random.randn(2,1)
parameters={"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
a,y_predt,caches=rnn_forward(x,a0,parameters)
print("a[4][1] = ", a[4][1])
print("a.shape = ", a.shape)

################################################################
                # RNN BACKPROP #
def rnn_backprop(da_next,cache):
    (a_next,a_prev,xt,parameters)=cache
    Wax=parameters["Wax"]
    Waa=parameters["Waa"]
    ba=parameters["ba"]
    by=parameters["by"]
    dtanh=(1-a_next**2)*da_next
    dWax=np.dot(dtanh,xt.T)
    dWaa=np.dot(dtanh,a_prev.T)
    dxt=np.dot(Wax.T,dtanh)
    da_prev=np.dot(Waa.T,dtanh)
    dba=np.sum(dtanh,axis=1,keepdims=1)
    gradient={"dxt":dxt,"da_prev":da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    return gradient

def rnn_backward(da,caches):
    (cache,x)=caches
    (a_next,a_prev,xt,parameters)=caches[0]
    n_a,m,T_x=da.shape
    ####ye mene ni kiya
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    ####yaha tak
    for t in reversed(range(T_x)):
        gradient=rnn_backprop(da[:,:,t]+da_prev,caches[t])
        dxt,da_prevt,dWaxt,dWaat,dbat=gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
        dx[:,:,t]=dxt
        dWax+=dWaxt
        dWaa+=dWaat
        dba+=dbat
        
    da0=da_prevt
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba}
    return gradients