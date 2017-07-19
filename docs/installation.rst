This is the Installation Guide for the Graphs Dynamics Library
=====================================================

Ideally we want to install the library in a virtual enviroment

	The calls required are:
	
		* virtualenv <virtualenv_directory>
		* <virtualenv_directory>/bin/pip install pandas, networkx, sklearn, matplotlib
		
		* Download snaps from http://snap.stanford.edu/snappy/release/
			* Untar snaps with: tar zxvf snap-3.0.0-3.0-centos6.5-x64-py2.6.tar.gz
			* Move to the snap folder: cd snap-3.0.0-3.0-centos6.5-x64-py2.6
			* Copy the snap.py and _snap.so files to the packages folder of the virtual enviroment
				* cp _snap.py <virtualenv_directory>/lib/python2.7/site-packages/snap.py
				* cp _snap.so <virtualenv_directory>/lib/python2.7/site-packages/snap.py
				
		* Install the tag2hierachy library 
			* mkdir  <tag2hierarchy-directory>
			* cd <tag2hierarchy-directory>
			* git clone https://github.com/cesarali/Tag2Hierarchy.git
			* cd <tag2hierarchy-directory>/Tag2Hierarchy
			*  <virtualenv_directory>/bin/pip install -e .
			
		* Install graph_dynamics
			* mkdir <graph_dynamics-directory>
			* cd  <graph_dynamics-directory>
			* git clone https://github.com/ernaneluis/graph-dynamics.git
			* cd grahp-dynamics
			* <virtualenv_directory>/bin/pip install -e .
			
	In order to use the library the virtual enviroment should be activated with the command:
	
	source  <virtualenv_directory>/bin/activate 
	
	
