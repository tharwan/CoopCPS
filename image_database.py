import matplotlib.pyplot as plt
import datetime
from subprocess import Popen, PIPE
from sys import argv
import os


def normPath(path):
	# clean up
	if path[-1]=='/':
		path=path[:-1]
		
	path = os.path.expanduser(path)
	return path


def saveFiguesWithData(path, figure_list, data, prefix=""):
	from pyexif import ExifEditor
	
	path = normPath(path)

	# create path if necessariy
	if not os.path.exists(path):
		os.makedirs(path)

	date_str = datetime.datetime.now().isoformat().replace(':','-')
	
	if not(isinstance(data,str)):	
		
		repo_rev = Popen(['hg','identify','-n'], shell=False, stdout=PIPE).stdout.read()
		data['revision']=repo_rev.strip()
		data['file']=argv[0]

		keywords = []
		for name,value in data.iteritems():
				keywords.append("{}:{}".format(name,value))
	
		keywords = ";".join(keywords)

	else:
		keywords = data

	fig_names = []
	for name,fig in figure_list.iteritems():
		figname = "{}/{}_{}-{}.jpg".format(path,prefix,name,date_str) 
		fig_names.append("{}_{}-{}.jpg".format(prefix,name,date_str))
		fig.savefig(figname)
		E = ExifEditor(figname)

		if isinstance(data,str):
			for pair in data.split(';'):
				E.addKeyword(pair)
		else:
			for name,value in data.iteritems():
				E.addKeyword("{}:{}".format(name,value))
	fig_names = ";".join(fig_names)
	

	with open("{}/{}".format(path,'log.log'),'a+') as f:
		record = "\t".join((date_str,keywords,fig_names))
		f.write(record+'\n')

	pass
	
