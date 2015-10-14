#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
pi = np.pi
from numpy.random import rand
from numpy.random import randint
from mpi4py import MPI
from cython_utils import frac_neighbours, mean_neighbours
import networkx as nx

class Controller(object):
	def __init__(self,N_agents,Ri=2,R=25,U=10,comm_error=0.1,\
		min_gain=lambda: 0.0001,agentClass=None,budget=1,silent=False,turns=1000,G=None,\
		generate_graph=False,selfishness=lambda: 0.5):
		self.silent = silent
		self.comm = MPI.COMM_WORLD
		self.mpi_size = self.comm.Get_size()
		self.rank = self.comm.Get_rank()

		if G is None:
			if self.rank == 0:
				if generate_graph:
					G = nx.connected_watts_strogatz_graph(N_agents,4,0.5)
				else:
					G = nx.cycle_graph(N_agents)
		self._G = self.comm.bcast(G,root=0)

		if agentClass is None:
			agentClass = Agent

		R = float(R) * N_agents
		self._N = N_agents  # Number of global agents
		self._Ri = float(Ri)
		self._R = float(R)
		self._U = float(U)
		self.budget = float(budget)
		self._turns = turns
		

		# number of agents for each worker
		base = np.floor(self._N / float(self.mpi_size))
		
		# hangover
		leftover = self._N % self.mpi_size

		sizes = np.ones(self.mpi_size)*base
		sizes[:leftover]+=1

		offsets = np.zeros(self.mpi_size)
		offsets[1:] = np.cumsum(sizes)[:-1]
		
		
		self.mpi_gather_sizes = sizes 
		self.mpi_gather_offsets = offsets

		start = offsets[self.rank]
		end = start + sizes[self.rank]
		agent_idxs = np.arange(start,end)
		
		self.local_agents = [agentClass(idx=i,R=R,
			s=max([0,min([0.99,selfishness()])]),
			# s=0.9,
			controller=self,
			min_gain=min_gain(),
			turns=turns) for i in agent_idxs]
		
		if not(self.silent):
			print "number of agents in rank {}:{}\n{}".format(self.rank,\
				len(self.local_agents),agent_idxs)
			

		for agent in self.local_agents:
			# print agent.min_gain,agent.s
			agent.k = randint(10)+1
			agent.budget=self.budget/self._N
			agent.alpha = 0.3
			agent.omega = 3+rand()
			# agent.nn = [agent.idx-1,(agent.idx+1)%self._N]
			agent.nn = self._G.neighbors(agent._idx)

			

		self._comm_error = comm_error
		
		

		self._I = []
		self._Pout = []
		self._Us = []
		self._Rg = []
		self._n = 0  # Number of steps we did
		self.is_constrained = True
		self._global_P = [1]
		self._global_dP_list = [1]
		self._global_dP = 0
		self.calc_global_dP = False
		self.globalGameState = {0:np.zeros(self._N,dtype=np.int8)}



		pass


	@property
	def turns(self):
		return self._turns

	@property
	def global_P(self):
		return self._global_P[-1]

	@global_P.setter
	def global_P(self,P):
		self._global_P.append(P)
		if self.calc_global_dP:
			y = self._global_P[-self.trend_horizon:]
			x = np.arange(len(y))
			p = np.polyfit(x,y,1)
			self._global_dP = p[0]
			self._global_dP_list.append(p[0])
		

	@property
	def global_dP(self):
		return self._global_dP

	@property
	def N_agents(self):
		return self._N
	
	@property
	def comm_error(self):
		return self._comm_error

	@property
	def R(self):
		return self._R

	@property
	def Ri(self):
		return self._Ri

	@property
	def U(self):
		return self._U

	@property
	def I(self):
		return self._I[-1]

	@I.setter
	def I(self,I_new):
		self._I.append(I_new)

	@property
	def P_out(self):
		return self._Pout[-1]

	@P_out.setter
	def P_out(self,P_new):
		self._Pout.append(P_new)

	@property
	def Us(self):
		return self._Us[-1]

	@Us.setter
	def Us(self,U_new):
		self._Us.append(U_new)

	@property
	def Rg(self):
		return self._Rg[-1]

	@Rg.setter
	def Rg(self,R_new):
		self._Rg.append(R_new)

		
	def getGlobalR(self):
		local_R_vec = np.array([a.R for a in self.local_agents],dtype=np.float64)
		global_R_vec = np.zeros(self._N,dtype=np.float64)
		
		# all subgroups sync together
		# everyone transmits his parts and they get conected together
		self.comm.Allgatherv(local_R_vec,[global_R_vec,self.mpi_gather_sizes,\
			self.mpi_gather_offsets,MPI.DOUBLE])
		return global_R_vec

	def getAllResistors(self):
		local_vec = np.array([a.k for a in self.local_agents],dtype=np.int16)

		#TODO: apply error?

		global_vec = np.zeros(self._N,dtype=np.int16)

		self.comm.Allgatherv(local_vec,[global_vec,\
			self.mpi_gather_sizes,self.mpi_gather_offsets,MPI.INT16_T])
		return global_vec
	
	def getGlobalGameState(self):
		local_Game_vec = np.array([a.decision[self._n-1] for a in self.local_agents],dtype=np.int8)
		
		# we apply the communicatoin error at this point so everyone gets the same "error"
		if self.comm_error>0:
			
			rand_vec = np.random.rand(len(local_Game_vec))
			rand_messages = np.random.randint(-1,2,len(local_Game_vec))

			local_Game_vec[rand_vec<self.comm_error]=rand_messages[rand_vec<self.comm_error]


		global_Game_vec = np.zeros(self._N,dtype=np.int8)
		
		# all subgroups sync together
		# everyone transmits his parts and they get conected together
		self.comm.Allgatherv(local_Game_vec,[global_Game_vec,\
			self.mpi_gather_sizes,self.mpi_gather_offsets,MPI.INT8_T])
		self.globalGameState = {self._n:global_Game_vec}



	# @profile
	def update_physics(self):
		self._n += 1

		if self.rank == 0:
			global_n = self._n
		else:
			global_n = None

		global_n = self.comm.bcast(global_n,root=0)

		assert global_n == self._n
		
		R = self._Ri


		# this was the fastest version. Allreduce was slower
		C_global = 0
		C_local = sum((1/a.R for a in self.local_agents))

		C_global=self.comm.allreduce(C_local,op=MPI.SUM)
		R += 1/C_global

		self.Rg = R
		self.I = self._U / R
		self.Us = self._U - self._Ri * self.I
		

		P = 0
		P_local = 0
		U2 = self.Us**2
		
		for agent in self.local_agents:
			P = U2/agent.R
			agent.P = P
			P_local += P

		# get and sum up all local P values
		P_global = self.comm.allreduce(P_local,op=MPI.SUM)

		self.global_P = P_global

		assert self._n-1 in self.globalGameState
		gameState = self.globalGameState[self._n-1]
		

		message = dict()
		message['n'] = self._n
		message['sender'] = self
		message['neighbours'] = gameState
		message['allResistors'] = self.getAllResistors()
		message['distress'] = self.Us < self._U/2

		for agent in self.local_agents:
			# we tell the agents that it is time to update themselfs and what 
			# time n we have and what their neighbours did
			
			
			agent.update(message)

	
	def run(self):
		import time
		t = time.time()
		for i in range(self.turns):
			self.update_physics()
			self.getGlobalGameState()
		print "processing time: {}".format(time.time()-t)
		


	def collectAgents(self):
		if self.rank == 0:
			self._agents = self.local_agents

		for i in range(1,self.mpi_size):
			
			if self.rank == 0:
				remote_agents = self.comm.recv(source=i,tag=0)
				self._agents.extend(remote_agents)

			if self.rank == i:
				self.comm.send(self.local_agents,dest=0,tag=0)



	def __str__(self):
		return 'Controller'

	def __repr__(self):
		return 'Controller'

class Agent(object):
	def __init__(self,idx,R,controller,s,P=1e-10,min_gain=0.001,budget=0,turns=1):
		self.R0 = float(R)
		self._R = self.R0
		self.min_gain = min_gain

		self.budget = budget

		self._k = 1
		# self._k_que = deque([1])
		self._k_que = np.zeros(turns+2)
		self._k_que[0] = self._k
		
		self._P = P
		# self._P_que = deque([P])
		self._P_que = np.zeros(turns+1)
		self._P_que[0] = self._P

		# deltas
		self._exp_P = 1
		self._exp_k = 1

		self.s = s
		self._idx = idx

		self.controller = controller


		

		# game decisions
		self.decision = np.zeros(turns+1,dtype=np.int8)
		self._n = 0 # Number of steps we did
		
		self._nn = []
		self.nn_mask = np.zeros(controller.N_agents)
		pass

	@property
	def nn(self):
		return self._nn

	@nn.setter
	def nn(self,nn_new):
		self._nn = nn_new
		self.nn_mask = np.zeros(self.controller.N_agents,dtype=np.uint8)
		self.nn_mask[nn_new] = 1

	@property
	def idx(self):
		return self._idx

	@property
	def R(self):
		return self._R

	@property
	def k(self):
		return self._k

	@k.setter
	def k(self,k_new):
		self._k_que[self._n]=int(k_new)
		self._exp_k = k_new - self._k
		self._k = max(int(k_new),1)
		self._R = self.R0/k_new

	@property
	def P(self):
		return self._P

	@P.setter
	def P(self,P_new):
		self._P_que[self._n]=P_new
		self._exp_P = (P_new - self._P)/self._P
		self._P = P_new


	def expected_dP(self):
		return self._exp_P

	def delta_k(self):
		return self._exp_k

	# @profile
	def update(self,message):

		if message['n']>self._n and message['sender']==self.controller:
			self._n = message['n']
			n = self._n

			if self.expected_dP() > self.min_gain:

				# we do what we did before
				
				if self.delta_k() > 0:
					self.k = self.k + 1 # add appliance
					self.decision[n]=-1
					
				elif self.delta_k() < 0:
					if self.k > 1:
						self.k = self.k - 1
					else:
						self.k = self.k
	
					self.decision[n]=1

				elif self.delta_k()==0:
	
					self.decision[n]=0
					self.k = self.k			
				
			else:
				
				
				
				f = frac_neighbours(message['neighbours'],self.nn_mask)
			
				if (rand() > self.s or (f >= 0.5)):
					
					self.decision[n] = 1
					
					if self.k > 1:
						self.k = self.k - 1
					else:
						self.k = self.k
				else:
					# if both neighbours defected or we just feel like it today we defect
					if rand() < self.s: 
						# we are very selfish
						self.decision[n] = -1
						self.k = self.k + 1
					else:
						self.decision[n]=0
						self.k = self.k
		
		
		else:
			pass # we can ignore the message since we are allready up to date

	def __str__(self):
		return 'Agent '+str(self._idx)
