#!/usr/bin/env python
# -*- coding: UTF-8 -*-


from utilities import *
from image_database import saveFiguesWithData
from sys import argv, exit
from numpy import load
import argparse
import seaborn as sns
from os import path as pathtools

parser = argparse.ArgumentParser(description='process data from cluster')
parser.add_argument('file',help="what file to work on",nargs='*')
parser.add_argument('-m','--metadata',action='store_true',help="print only metadata")
parser.add_argument('-s','--save',action='store_true',help="save images to dropbox")
parser.add_argument('--save_only',action='store_true',help="save images to dropbox, do not show on screen")
parser.add_argument('-p','--save_path',help="override the dafault save path")
args = parser.parse_args()

for filename in args.file:

	f = load(filename)
	print filename
	meta = str(f['metadata'])
	meta = meta.replace(';','\n')
	print meta

	if args.metadata:
		exit()

	plt.close("all")

	figs = {}


	fig_k = plt.figure()
	plot_appliances_aggregate(f['appliances'],f['t'])
	figs['appliances']=fig_k

	fig_behaviour = plt.figure(figsize=(12,6))
	matrix = plot_behavior(f['beh_matrix'])
	figs['behavior']=fig_behaviour

	agent_power = plt.figure()
	plot_agent_power(f['P_all'],f['t'][1:])
	figs['agent_power']=agent_power

	overall_power = plt.figure()
	plot_power_usage(f['P_global'],f['t'][1:])
	figs['overall_power']=overall_power

	plt.figure()
	plot_appl_matrix(f['appliances'])

	plt.figure()
	matrix = f['appliances']
	app = downsample(matrix)

	time = downsample(f['t'])

	sns.tsplot(app,time=time, err_style="unit_traces", err_palette=sns.dark_palette("crimson", len(app)), color="k");
	plt.xlabel('time')
	plt.ylabel('app')

	plt.figure()
	s = f['selfish']
	plt.plot(s)
	plt.ylim([0,1])
	plt.xlabel('agent')
	plt.ylabel('selfishness')	

	meta = str(f['metadata'])
	meta_dict = {pair.split(':')[0]:pair.split(':')[1]  for pair in meta.split(';')}

	P_max = float(meta_dict['U'])**2/float(meta_dict['Ri'])/4
	p_matrix = f['P_all']
	sum_P = np.mean(p_matrix,axis=1)
	p_equal = P_max/float(p_matrix.shape[0])
	print "p_equal", p_equal, "P_max", P_max, "ptp", np.ptp(sum_P-p_equal), "gini",gini_coeff(sum_P)

	if args.save or args.save_only:
		path = args.save_path
		saveFiguesWithData(path, figs, str(f['metadata']),prefix=pathtools.basename(filename)[:-4])

	if not(args.save_only):
		plt.show()
