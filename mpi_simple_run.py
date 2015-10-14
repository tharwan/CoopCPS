#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import mpi_resistor_model as resistor_model
from functools import partial
import numpy as np
import networkx as nx

steps = 40000
selfishness = lambda: np.random.rand()
gain = lambda: 0.0005 

con = resistor_model.Controller(100, Ri=2, R=200, U=10, comm_error=0.01,
	overcurrent=4.5,min_gain=gain,turns=steps,generate_graph=True,
,selfishness=selfishness)
con.is_constrained=False
con.run()
con.collectAgents()


if con.rank == 0:
	print "computation finished."	
	import utilities

	data = {'steps':steps,'Ri':con.Ri,'R':con.R/con._N,'U':con.U,
		'comm_error':con.comm_error,'overcurrent':con.overcurrent,
		'min_gain':0.0005,'constrained':con.is_constrained,
		'N_agents':con.N_agents,'horizon':con.trend_horizon}

	utilities.save(con,data,'test.npz')
