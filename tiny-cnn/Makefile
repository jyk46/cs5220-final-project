build: configure
	./waf -v build

configure:
	./waf configure --TBB_ROOT=/usr/local/intel/tbb --BOOST_ROOT=/export/apps/boost-1.59.0-gcc-4.9.2/include

configure-offload:
	./waf configure --OFFLOAD --AVX --OMP --TBB_ROOT=/usr/local/intel/tbb --BOOST_ROOT=/export/apps/boost-1.59.0-gcc-4.9.2/include --TARGET=offload

configure-omp:
	./waf configure --AVX --OMP --TBB_ROOT=/usr/local/intel/tbb --BOOST_ROOT=/export/apps/boost-1.59.0-gcc-4.9.2/include --TARGET=omp

configure-tbb:
	./waf configure --AVX --TBB --TBB_ROOT=/usr/local/intel/tbb --BOOST_ROOT=/export/apps/boost-1.59.0-gcc-4.9.2/include --TARGET=tbb

configure-none:
	./waf configure --AVX --TBB_ROOT=/usr/local/intel/tbb --BOOST_ROOT=/export/apps/boost-1.59.0-gcc-4.9.2/include --TARGET=none
