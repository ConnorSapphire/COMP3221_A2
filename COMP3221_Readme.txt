Coding Environment:
	Language:
		python version 3.12.3

	Default Packages:
		sys
		socket
		threading
		json
		pickle
		time
		random

	Installed Packages:
		pip version 24.0
		torch version 2.2.2
		pandas version 2.1.4
		numpy version 1.26.2
		matplotlib version 3.8.4



Run Program:
	Info:
		This program is meant to be imitating a server communicating with clients.
		Therefore the clients and server must each be run on separate terminals. The
		server must be the first program run in order for the program to execute
		correctly. There can only be 1 server, but there can be 0-5 clients. Please
		use the commands below to run the program in the modes that you want.

		Subsampling means that of the local models, only the selected amount can be
		aggregated into the global model. The models which get aggregated are chosen
		randomly.
		
		Mini-batch means that the available training data of a client is divided into
		subsets which are then used for calculating the MSE and performing gradient
		descent for the current epoch. This can lead to convergence in fewer epochs.
	
	Run Server:
		No Subsampling:	
			$ python COMP3221_FLServer.py 6000 0

		N-Subsampling:
			$ python COMP3221_FLServer.py 6000 <N>
			
			where <N> is the number of local models to be aggregated into the global
			model each iteration.
	
	Run Client:
		Gradient Descent:
			$ python COMP3221_FLClient.py <port> <client_id> 0
			
			where <port> is the corresponding port for the <client_id>. <client_id> is
			of the form "client<n>" where <n> is the number representing the client, in
			the range (1-5), and <port> equals 600<n>.

		Mini-Batch Gradient Descent:
			$ python COMP3221_FLClient.py <port> <client_id> 1

			where <port> is the corresponding port for the <client_id>. <client_id> is
			of the form "client<n>" where <n> is the number representing the client, in
			the range (1-5), and <port> equals 600<n>.



Reproduce Experimental Results:
	NB: Depending on the terminal used, the "./" in the following commands may
	need to be removed.

	Run 5 Gradient Descent Clients with a Non-Subsampling Server:
		$ ./BatchRunRegular.bat

	Run 5 Gradient Descent Clients with a 3-Subsampling Server:
		$ ./BatchRunSubsample.bat

	Run 5 Mini-Batch Clients with a Non-Subsampling Server:
		$ ./BatchRunMiniBatch.bat

	Run 5 Mini-Batch Clients with a 3-Subsampling Server:
		$ ./BatchRunSubsampleMiniBatch.bat
	
	Run 3 Gradient Descent and 2 Mini-Batch Clients with a Non-Subsampling Server:
		$ ./BatchRunMix.bat

	Run 3 Gradient Descent and 2 Mini-Batch Clients with a 3-Subsampling Server:
		$ ./BatchRunSubsampleMix.bat
	
	Run 1 Gradient Descent Client with a Non-Subsampling Server:
		$ ./BatchTest.bat

	Run 3 Gradient Descent Client with a Non-Subsampling Server:
		$ ./BatchTest3.bat

	Run 2 Gradient Descent and 1 Mini-Batch Client with a 2-Subsampling Server:
		$ ./BatchTestMini.bat
