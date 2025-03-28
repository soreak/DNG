From ann-benchmarks

RUN git clone https://github.com/soreak/DNG
RUN cd DNG && python3 setup.py install
RUN python3 -c 'import DNG'