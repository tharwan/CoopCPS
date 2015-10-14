#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from os import path
import sys
from networkx import to_numpy_matrix
import resource


def askToSaveFig(name,fig):
	answer = raw_input("Save as {name}? (y/n)".format(name=name))
	if answer=='Y' or answer=='Yes' or answer=='y':
		fig.savefig(name)
	return

def getMetaDict(f):
	meta = str(f['metadata'])
	meta_dict = {pair.split(':')[0]:pair.split(':')[1]  for pair in meta.split(';')}
	return meta_dict

def getPmax(f):
	meta_dict = getMetaDict(f)
	P_max = float(meta_dict['U'])**2/float(meta_dict['Ri'])/4
	return P_max

def typical_length(sig):
	sig = sig.astype(float)
	sig -= sig.mean()
	sig /= sig.ptp()
	ffted = np.fft.fft(sig)
	dom = np.argmax(ffted[:len(ffted)/2])
	freq = np.fft.fftfreq(len(sig))
	
	if dom==0:
		typ = np.inf
	else:
		typ = 1/freq[dom]

	return typ,abs(ffted[dom])*2/len(sig)


def compose_beh_matrix(con):
	matrix = []
	for idx,agent in enumerate(con._agents):
		decisions = agent.decision
		if isinstance(decisions[-1],basestring):
			decisions = np.array([1 if d=='c' else (d=='0')-1 for d in decisions])  #"c"==1 "d"==-1 "0"==0
		else:
			decisions = np.array(decisions)
		
		matrix.append(decisions)
		
	return np.array(matrix)

def collect_appliances(con):
	appliances = []
	for idx,agent in enumerate(con._agents):
		appliances.append(agent._k_que)

	t = np.arange(len(con._agents[0]._k_que))
	
	appliances = np.array(appliances)
	return appliances,t

def collect_power(con):
	P_list = []
	for idx,agent in enumerate(con._agents):
		P_list.append(agent._P_que)

	P_all = np.array(P_list)
	time = np.arange(len(con._agents[0]._P_que))
	return P_all,time

def collect_exp_P(con):

	P_list = []
	for idx,agent in enumerate(con._agents):
		P_list.append(agent._exp_P_que)

	P_all = np.array(P_list)
	time = np.arange(len(con._agents[0]._exp_P_que))

	return P_all,time

def save(con,data,filename=None):
	
	from subprocess import Popen, PIPE
	from sys import argv
	import datetime
	from os import path

	kwargs = {}
	kwargs['beh_matrix'] = compose_beh_matrix(con)
	appliances,t = collect_appliances(con)
	kwargs['appliances'] = appliances
	kwargs['t'] = t
	P_all,t = collect_power(con)
	kwargs['P_all'] = P_all
	kwargs['P_global'] = np.array(con._global_P)
	kwargs['selfish'] = np.array([a.s for a in con._agents])
	kwargs['graph'] = np.array(to_numpy_matrix(con._G))

	if hasattr(con._agents[0],'_exp_P_que'):
		P_exp,t = collect_exp_P(con)
		kwargs['P_exp'] = P_exp

		
	data['file']=argv[0]

	if filename is None:
		date_str = datetime.datetime.now().isoformat().replace(':','-')
		filename = '{}-{}.npz'.format(data['file'][:-3],date_str)

	keywords = []
	for name,value in data.iteritems():
			keywords.append("{}:{}".format(name,value))

	keywords = ";".join(keywords)

	kwargs['metadata'] = keywords

	if sys.platform != 'darwin':
		filename = "/".join((path.expandvars('$WRKDIR'),filename))

	root,extension = path.splitext(filename)


	test_filename = filename
	i = 0
	while path.exists(test_filename):
		test_filename = "{}_{}{}".format(root,i,extension)
		i+=1
	filename=test_filename
	
	print 'save as ',filename
	np.savez_compressed(filename,**kwargs)





def downsample(x):
	def ds_one_dim(x):
		if len(x)<1000:
			return x
		else:
			R = int(np.floor(len(x)/1000.0))
			slices = len(x)/R  # note this is an integer division
			return x[:slices*R].reshape(-1, R).mean(axis=1)  # we ignore the overhang
	
	x = np.array(x)
	if x.ndim>1:
		n = x.shape[0]
	else:
		return ds_one_dim(x)
	ds_x = []
	for i in range(n):
		ds_x.append(ds_one_dim(x[i,:]))
	return np.array(ds_x)
	
		 

def plot_appliances_aggregate(x,time=None):
	import seaborn as sns
	if isinstance(x,np.ndarray):
		appliances = x
	else:
		appliances,time = collect_appliances(x)
	time = downsample(time)
	appliances = downsample(appliances)
	
	sns.tsplot(appliances,time=time)
	plt.xlabel("steps")
	plt.ylabel("appliances")

def plot_appl_matrix(x):

	if isinstance(x,np.ndarray):
		matrix = x
	else:
		matrix,t = collect_appliances(x)

	high = np.max(matrix)
	print high

	image = matrix/float(high)*255
	image = image.astype(np.uint8)


	image = np.dstack([image,image,image])


	plt.imshow(image,aspect="auto",interpolation='nearest')

	plt.xlabel("steps")
	plt.ylabel("agent")	
	plt.grid('off')
	return image
	
def plot_behavior(x):

	if isinstance(x,np.ndarray):
		matrix = x
	else:
		matrix = compose_beh_matrix(x)

	image_r = np.zeros(matrix.shape,dtype=np.uint8)
	image_g = np.zeros(matrix.shape,dtype=np.uint8)
	image_b = np.zeros(matrix.shape,dtype=np.uint8)

	# make cooperation white
	image_r[matrix==1]=255
	image_g[matrix==1]=255
	image_b[matrix==1]=255

	# make defection red
	image_r[matrix==-1]=165
	image_g[matrix==-1]=80
	image_b[matrix==-1]=80

	# make don‘t care black
	image_r[matrix==0]=0
	image_g[matrix==0]=0
	image_b[matrix==0]=0

	# make don‘t care black
	image_r[matrix==-2]=0
	image_g[matrix==-2]=255
	image_b[matrix==-2]=255

	image = np.dstack([image_r,image_g,image_b])


	plt.imshow(image,aspect="auto",interpolation='nearest')

	plt.xlabel("steps")
	plt.ylabel("agent")	
	plt.grid('off')
	return image

def plot_power_usage(x,time=None):

	if isinstance(x,np.ndarray):
		P = x
	else:
		time = np.arange(len(x._global_P))
		time = downsample(time)
		P = downsample(x._global_P)
	plt.plot(time,P)

	plt.xlabel("steps")
	plt.ylabel("power")

def plot_agent_power(x,time=None):
	import seaborn as sns
	if isinstance(x,np.ndarray):
		P_all=x
	else:
		P_all,time = collect_power(x)

	P_all = downsample(P_all)
	time = downsample(time)

	sns.tsplot(P_all,time=time, err_style="unit_traces", err_palette=sns.dark_palette("crimson", len(P_all)), color="k");

	plt.xlabel("steps")
	plt.ylabel("P")

	return P_all

def plot_agent_expected(x,time=None):
	import seaborn as sns
	if isinstance(x,np.ndarray):
		P_all=x
	else:
		P_all,time = collect_power(x)

	P_all = downsample(P_all)
	time = downsample(time)
	
	x = downsample(x)
	sns.tsplot(P_all,time=x, err_style="unit_traces", err_palette=sns.dark_palette("crimson", len(P_all)), color="k");

	plt.xlabel("steps")
	plt.ylabel("expected_dP")

	return P_all

def gini_coeff(x): 
	x = x.copy()
	x += x.min()
	xsort = np.sort(x) 
	l = float(len(x))
	return 2*np.sum(xsort*np.arange(1,l+1))/(xsort.sum()*l) - (l+1)/l

def using(point=""):
	usage=resource.getrusage(resource.RUSAGE_SELF)
	return '''%s: usertime=%s systime=%s mem=%s mb
		   '''%(point,usage[0],usage[1],
				(usage[2]*resource.getpagesize())/1000000.0 )

def reduce_multiple_measurements(x_points,y_points,func=np.median):
	point_dict = dict()
	for x,y in zip(x_points,y_points):
		if x in point_dict:
			point_dict[x].append(y)
		else:
			point_dict[x]=[y]


	new_x = []
	new_y = []
	for key in sorted(point_dict.keys()):
		new_x.append(key)
		new_y.append(func(point_dict[key]))


	return np.array(new_x),np.array(new_y)
