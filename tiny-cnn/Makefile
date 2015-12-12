offload: configure-offload
	./waf build

omp: configure-omp
	./waf build

tbb: configure-tbb
	./waf build

none: configure-none
	./waf build

configure-offload:
	./waf configure --OFFLOAD --AVX --OMP --TBB_ROOT=/usr/local/intel/tbb/include --BOOST_ROOT=/export/apps/boost-1.59.0-gcc-4.9.2/include --TARGET=offload

configure-omp:
	./waf configure --AVX --OMP --TBB_ROOT=/usr/local/intel/tbb/include --BOOST_ROOT=/export/apps/boost-1.59.0-gcc-4.9.2/include --TARGET=omp

configure-tbb:
	./waf configure --AVX --TBB --TBB_ROOT=/usr/local/intel/tbb/include --BOOST_ROOT=/export/apps/boost-1.59.0-gcc-4.9.2/include --TARGET=tbb

configure-none:
	./waf configure --AVX --TBB_ROOT=/usr/local/intel/tbb/include --BOOST_ROOT=/export/apps/boost-1.59.0-gcc-4.9.2/include --TARGET=none
