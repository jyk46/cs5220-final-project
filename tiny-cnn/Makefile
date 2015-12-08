omp: configure-omp
	./waf build && qsub omp.pbs

tbb: configure-tbb
	./waf build && qsub tbb.pbs

none: configure-none
	./waf build && qsub cnn.pbs

configure-omp:
	./waf configure --OMP --TBB_ROOT=/usr/local/intel/tbb/include --BOOST_ROOT=/export/apps/boost-1.59.0-gcc-4.9.2/include --TARGET=omp

configure-tbb:
	./waf configure --AVX --TBB --TBB_ROOT=/usr/local/intel/tbb/include --BOOST_ROOT=/export/apps/boost-1.59.0-gcc-4.9.2/include --TARGET=tbb

configure-none:
	./waf configure --AVX --TBB_ROOT=/usr/local/intel/tbb/include --BOOST_ROOT=/export/apps/boost-1.59.0-gcc-4.9.2/include --TARGET=cnn
