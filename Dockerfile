FROM ubuntu:16.04

RUN	apt-get update -y

RUN	apt-get install -y curl && \
	apt-get install -y git && \
	apt-get install -y less && \
	apt-get install -y vim && \
	apt-get install -y vim-common && \
	apt-get install -y tar && \
	apt-get install -y zip && \
	apt-get install -y unzip

RUN	apt-get update

RUN	apt-get install -y build-essential && \
 	apt-get install -y apt-utils && \
	apt-get install -y automake && \
	apt-get install -y cmake && \
	apt-get install -y gcc && \
	apt-get install -y gcc-4.9 && \
	apt-get install -y g++ && \
	apt-get install -y g++-4.9
	
	

RUN	apt-get install -y wget

RUN	apt-get install -y libblas-dev && \
	apt-get install -y liblapack-dev

RUN	apt-get install -y libreadline-dev && \
	apt-get install -y readline-common 
	
	

RUN	curl -R -O http://www.lua.org/ftp/lua-5.3.4.tar.gz && \
	tar zxf lua-5.3.4.tar.gz && \
	cd lua-5.3.4 && \
	make linux && \
	make install

RUN	apt-get install -y libjpeg-dev && \
	apt-get install -y libpng12-dev
