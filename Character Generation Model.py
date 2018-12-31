import numpy as np

def softmax(x):
    ex=np.exp(x-np.max(x))
    return ex/ex.sum(axis=0)

def sigmoid(x):
    return 1/(1+np.exp(-x))
	
							#####CHARACTER LANGUAGE MODEL######
		
data=open("names.txt","r").read()
data=data.lower()
chars=set(data) 
labeling={ch:i for i,ch in enumerate(sorted(chars))}
unlabeling={i:ch for i,ch in enumerate(sorted(chars))}

#######################################################

def sample(parameters,labeling):
    (Wax,Waa,Wya,by,ba)=parameters["Wax"],parameters["Waa"],parameters["Wya"],parameters["by"],parameters["ba"]
    n_a,vocab=Wax.shape
    x=np.zeros((vocab,1))
    a_prev=np.zeros((n_a,1))
    counter=0
    index=[]
    idt=-1
    while(counter!=50 and idt!=labeling['\n']):
         a=np.tanh(np.dot(Wax,x)+np.dot(Waa,a_prev)+ba)
         y=softmax(np.dot(Wya,a)+by)
         idt=np.random.choice(list(range(vocab)),p=y.ravel())
         index.append(idt)
         x=np.zeros((vocab,1))
         x[idt]=1
         a_prev=a
         counter+=1
    if (counter==50):
        index.append(labeling['\n'])
    return index

########################################################

def clip(gradients,maxval):
    dWax,dWaa,dWya,dba,dby=gradients['dWax'], gradients['dWaa'], gradients['dWya'], gradients['dba'], gradients['dby']
    for gradient in [dWax, dWaa, dWya, dba, dby]:
        np.clip(gradient, -maxval, maxval, out=gradient)
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "dba": dba, "dby": dby}
    return gradients
	
#######################################################

def rnn_forward(X,Y,a0,parameters,vocab=27):
    x,a,yt_hat={},{},{}
    a[-1]=np.copy(a0)
    loss=0
    for t in range(len(X)):
        x[t]=np.zeros((vocab,1))
        if (X[t]!=None):
            x[t][X[t]]=1
        a[t],yt_hat[t]=rnn_cell_forward(x[t],a[t-1],parameters)
        loss-=np.log(yt_hat[t][Y[t]])
    cache=(yt_hat,a,x)
    return loss,cache
	
############################################

def rnn_cell_forward(x,a_prev,parameters):
    Wax,Waa,Wya,ba,by=parameters['Wax'],parameters['Waa'],parameters['Wya'],parameters['ba'],parameters['by']
    a_next=np.tanh(np.dot(Wax,x)+np.dot(Waa,a_prev)+ba)
    yt=softmax(np.dot(Wya,a_next)+by)
    return a_next,yt
	
#############################################

def rnn_backward(X,Y,parameters,cache):
    yt,a,x=cache
    gradient={}
    Wax,Waa,Wya,ba,by=parameters["Wax"],parameters["Waa"],parameters["Wya"],parameters["ba"],parameters["by"]
    gradient['dWax'],gradient['dWaa'],gradient['dWya'],gradient['dba'],gradient['dby'],gradient["da_prev"]=np.zeros_like(Wax),np.zeros_like(Waa),np.zeros_like(Wya),np.zeros_like(ba),np.zeros_like(by),np.zeros_like(a[0])
#    print("**rnn_back***Wax**%s, *** dWax %s****"%(str(parameters["Wax"].shape),str(gradient["dWax"].shape)))
    for t in reversed(range(len(X))):
        dy=np.copy(yt[t])
        dy[Y[t]]-=1
        gradient=rnn_step_backward(x[t],dy,a[t],a[t-1],gradient,parameters)
    return gradient,a
	
#############################################

def rnn_step_backward(x,dy,a,a_prev,gradient,parameters):
    gradient["dWya"]+=np.dot(dy,a.T)#n_y,n_a
    gradient["dby"]+=dy#n_y,1
    da_next=np.dot(parameters["Wya"].T,dy)+gradient["da_prev"]
    dtanh=(1-a**2)*da_next
    gradient["dWax"]+=np.dot(dtanh,x.T)
    gradient["dWaa"]+=np.dot(dtanh,a_prev.T)
    gradient["dba"]+=dtanh
    gradient["da_prev"]=np.dot(parameters["Waa"].T,dtanh)
    return gradient
	
###############################################

def update_parameters(parameters,gradient,lr):
#    print("shape****Wax****\n ",parameters["Wax"].shape)
#    print("shape****dWax****\n ",gradient["dWax"].shape)
    parameters["Wax"]+= -lr*gradient["dWax"]
    parameters["Waa"]+= -lr*gradient["dWaa"]
    parameters["Wya"]+= -lr*gradient["dWya"]
    parameters["ba"]+= -lr*gradient["dba"]
    parameters["by"]+= -lr*gradient["dby"]
    return parameters
	
###############################################

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    gradient,a=rnn_backward(X,Y,parameters,cache)
    gradient=clip(gradient,5)
    parameters=update_parameters(parameters,gradient,learning_rate)
    return loss,gradient,a[len(X)-1]
	
###############################################

def model(data,labeling,unlabeling,num_iter=70000,n_a=50,vocab=27,samples=7):
    Wax=np.random.randn(n_a,vocab)*0.1
    Waa=np.random.randn(n_a,n_a)*0.1
    Wya=np.random.randn(vocab,n_a)*0.1
    ba=np.random.randn(n_a,1)*0.1
    by=np.random.randn(vocab,1)*0.1
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba,"by": by}
    
    # initializing intial loss
    loss=-np.log(1.0/vocab)*samples
    with open("dinos.txt") as f:
        ex=f.readlines()
    ex=[x.lower().strip() for x in ex]
    np.random.shuffle(ex)
    a_prev=np.zeros((n_a,1))
    for j in range(num_iter):
        ind=j%len(ex)
        X=[None]+[labeling[ch] for ch in ex[ind]]
        Y=X[1:]+[labeling['\n']]
        curr_loss,gradient,a_prev=optimize(X,Y,a_prev,parameters)
        #smoothing the loss###i dont now why the hell are we doing this
        if j%2000==0:
            print("iteration:%d, Loss:%f"%(j,curr_loss))
            for name in range(samples):
               sampled_ind=sample(parameters,labeling) 
               txt="".join(unlabeling[i] for i in sampled_ind)
               txt=txt[0].upper() +txt[1:]
               print ('%s' % (txt, ), end='')#print(txt)
            print("************ \n ")
    return parameters
parameters=model(data,labeling,unlabeling)
