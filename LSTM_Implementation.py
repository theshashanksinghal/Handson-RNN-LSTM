import numpy as np

def softmax(x):
    ex=np.exp(x-np.max(x))
    return ex/ex.sum(axis=0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

##########################################################################################
	
                    # LSTM FORWARD PROP #
					
def lstm_cell(xt,a_pre,c_pre,parameters):
    Wf=parameters["Wf"]
    bf=parameters["bf"]
    Wu=parameters["Wu"]
    bu=parameters["bu"]
    Wo=parameters["Wo"]
    bo=parameters["bo"]
    Wc=parameters["Wc"]
    bc=parameters["bc"]
    Wy=parameters["Wy"]
    by=parameters["by"]
    
    n_x,m=xt.shape
    n_a,m=a_pre.shape
    concat=np.zeros(((n_x+n_a),m))
    concat[:n_a,:]=a_pre
    concat[n_a:,:]=xt       #instead of  writing all the 3 lines , use np.vstack((a_prev,xt))
    rf=sigmoid(np.dot(Wf,concat)+bf)
    ru=sigmoid(np.dot(Wu,concat)+bu)
    ro=sigmoid(np.dot(Wo,concat)+bo)
    c_tilt=np.tanh(np.dot(Wc,concat)+bo)
    ct=np.multiply(rf,c_pre)+np.multiply(ru,c_tilt)
    a_next=ro*np.tanh(ct)
    
    yt_pred=softmax(np.dot(Wy,a_next)+by)
    cache=(a_next,ct,a_pre,c_pre,rf,ru,ro,c_tilt,xt,parameters)
    return a_next,ct,yt_pred,cache

def lstm_forward(x,a0,parameters):
    caches=[]
    n_x,m,T_x=x.shape
    n_a,m=a0.shape
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    c_next=np.zeros((a0.shape))
    a_next=a0

    for t in range(T_x):
        a_nex,c_nex,yt,cache=lstm_cell(x[:,:,t],a_next,c_next,parameters)
        a[:,:,t]=a_nex
        c[:,:,t]=c_nex
        y[:,:,t]=yt
        caches.append(cache)
    caches=(caches,x)
    return a,y,c,caches
    
################################################################################

								# LSTM BACK PROP #
def lstm_back_cell(dat,dct_forwarded,cache):
    (a_next,ct,a_pre,c_pre,rf,ru,ro,c_tilt,parameters)=cache
    n_a,m=a_next.shape
    
    Wf=parameters["Wf"]
    bf=parameters["bf"]
    Wu=parameters["Wu"]
    bu=parameters["bu"]
    Wo=parameters["Wo"]
    bo=parameters["bo"]
    Wc=parameters["Wc"]
    bc=parameters["bc"]
    Wy=parameters["Wy"]
    by=parameters["by"]
    
    dct=dat*ro*(1-np.tanh(ct)**2)+ dct_forwarded
    dro=np.tanh(ct)*dat*ro*(1-ro)
    dc_tilt=ru*(1-c_tilt**2)*dct
    dru=dct*c_tilt*(1-ru)*ru
    drf=dct*c_pre*(1-rf)*rf
    dc_pre=dct*rf
    da_pre=np.dot(Wf[:,:n_a].T,drf)+np.dot(Wu[:,:n_a].T,dru)+np.dot(Wo[:,:n_a].T,dro)+np.dot(Wc[:,:n_a].T,dc_tilt)
    dxt=np.dot(Wf[:,n_a:].T,drf)+np.dot(Wu[:,n_a:].T,dru)+np.dot(Wo[:,n_a:].T,dro)+np.dot(Wc[:,n_a:].T,dc_tilt)
    dWc=np.dot(dc_tilt,np.vstack((a_pre,xt)).T)
    dWu=np.dot(dru,np.vstack((a_pre,xt)).T)
    dWf=np.dot(drf,np.vstack((a_pre,xt)).T)
    dWo=np.dot(dro,np.vstack((a_pre,xt)).T)
    dbc=np.sum(dc_tilt,axis=1,keepdims=True)
    dbu=np.sum(dru,axis=1,keepdims=True)
    dbf=np.sum(drf,axis=1,keepdims=True)
    dbo=np.sum(dro,axis=1,keepdims=True)
    grad={"dxt":dxt,"da_pre":da_pre,"dc_pre":dc_pre,"dWc":dWc,"dWu":dWu,"dWf":dWf,"dWo":dWo,"dbc":dbc,"dbu":dbu,"dbf":dbf,"dbo":dbo}
    return grad

def lstm_back(da,caches):
    (caches,x)=caches
    n_x,m=x[:,:,t].shape
    n_a,m,T_x=da.shape
    dWc=np.zeros((n_a,n_a+n_x))
    dWu=np.zeros((n_a,n_a+n_x))
    dWf=np.zeros((n_a,n_a+n_x))
    dWo=np.zeros((n_a,n_a+n_x))
    dbc=np.zeros((n_a,1))
    dbu=np.zeros((n_a,1))
    dbf=np.zeros((n_a,1))
    dbo=np.zeros((n_a,1))
    dx=np.zeros((n_x,m,T_x))
    da_pre=np.zeros((n_a,m))
    dc_prev=np.zeros((n_a,m))
    da0=np.zeros((n_a,m))
    for t in reversed(range(T_x)):
        gradient=lstm_back_cell(da[:,:,t]+da_prev,dc_prev,cache)
        dx[:,:,t]=gradient["dxt"]
        dWc+=gradient["dWc"]
        dWu+=gradient["dWu"]
        dWf+=gradient["dWf"]
        dWo+=gradient["dWo"]
        dbc+=gradient["dbc"]
        dbu+=gradient["dbu"]
        dbf+=gradient["dbf"]
        dbo+=gradient["dbo"]
    da0=gradient["da_pre"]
    gradients=["da0":da0,"dx":da,"dWc":dWc,"dWu":dWu,"dWf":dWf,"dWo":dWo,"dbc":dbc,"dbu":dbu,"dbf":dbf;"dbo":dbo]
    return gradients